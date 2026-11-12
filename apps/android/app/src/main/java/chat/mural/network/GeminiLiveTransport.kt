package chat.mural.network

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioDeviceCallback
import android.media.AudioDeviceInfo
import android.media.AudioFocusRequest
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import chat.mural.R
import java.io.IOException
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.sqrt
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put
import okhttp3.CookieJar
import okhttp3.HttpUrl
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString

/** Gemini Live uses JSON over WebSocket and PCM audio, unlike the OpenAI WebRTC transport. */
class GeminiLiveTransport(
    context: Context,
    private val scope: CoroutineScope,
    private val credentials: CredentialStore,
    private val client: OkHttpClient = defaultClient(),
    private val endpoint: HttpUrl = LIVE_ENDPOINT,
) : VoiceTransport {
    override var onEvent: ((JsonObject) -> Unit)? = null
    override var onFailure: ((String) -> Unit)? = null
    override var onLevels: ((Double, Double) -> Unit)? = null

    private val applicationContext = context.applicationContext
    private val audioManager = applicationContext.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private val generation = AtomicLong(0)
    private val lock = Any()
    private val retiredAttempts = ArrayDeque<Attempt>()
    private val audioScope = CoroutineScope(SupervisorJob() + AUDIO_DISPATCHER)
    private val ioScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    @Volatile private var activeAttempt: Attempt? = null
    suspend fun connect(instructions: String, history: JsonArray, language: String?) {
        val key = withContext(Dispatchers.IO) { credentials.read() } ?: throw GeminiException.MissingKey
        var attempt: Attempt? = null
        try {
            val created = withContext(AUDIO_DISPATCHER) {
                synchronized(lock) {
                    activeAttempt?.let(retiredAttempts::addLast)
                    activeAttempt = null
                    generation.incrementAndGet()
                }
                drainRetiredAttempts()
                val next = synchronized(lock) {
                    Attempt(
                        id = generation.incrementAndGet(),
                        previousAudioMode = audioManager.mode,
                        previousSpeakerphone = if (Build.VERSION.SDK_INT < 31) audioManager.isSpeakerphoneOn else false,
                        apiKey = key,
                        instructions = instructions,
                        history = history,
                        language = language,
                    ).also {
                        activeAttempt = it
                    }
                }
                attempt = next
                configureAudio(next)
                prepareAudio(next)
                next.socket = openSocket(next, resumed = false, resumptionHandle = null)
                next
            }
            withTimeout(READY_TIMEOUT_MILLISECONDS) { created.started.await() }
            created.scopeCompletion = scope.coroutineContext[Job]?.invokeOnCompletion {
                audioScope.launch { cleanupIfCurrent(created) }
            }
        } catch (_: kotlinx.coroutines.TimeoutCancellationException) {
            attempt?.let { withContext(kotlinx.coroutines.NonCancellable) { cleanupIfCurrent(it) } }
            throw GeminiLiveException(applicationContext.getString(R.string.error_transport_timeout))
        } catch (error: CancellationException) {
            attempt?.let { withContext(kotlinx.coroutines.NonCancellable) { cleanupIfCurrent(it) } }
            throw error
        } catch (error: Exception) {
            attempt?.let { withContext(kotlinx.coroutines.NonCancellable) { cleanupIfCurrent(it) } }
            throw error
        }
    }

    override fun send(event: JsonObject): Boolean {
        val attempt = activeAttempt ?: return false
        if (!isCurrent(attempt) || !attempt.started.isCompleted || attempt.reconnecting.get()) return false
        audioScope.launch {
            if (!isCurrent(attempt)) return@launch
            val message = GeminiLiveProtocol.command(event, attempt.language) ?: return@launch
            if (!sendNow(attempt, message) && !attempt.closing.get()) {
                fail(attempt, applicationContext.getString(R.string.error_transport_channel_closed))
            }
        }
        return true
    }

    override fun mute(muted: Boolean) {
        val attempt = activeAttempt ?: return
        attempt.muted.set(muted)
        if (muted) attempt.lastInputLevel = 0.0
        emitLevels(if (muted) 0.0 else attempt.lastInputLevel, attempt.lastOutputLevel, attempt)
    }

    override fun close() {
        val attempt = activeAttempt ?: return
        if (!attempt.closing.compareAndSet(false, true)) return
        attempt.socket?.close(1000, null)
        val seconds = ((SystemClock.elapsedRealtime() - attempt.startedAtMilliseconds).coerceAtLeast(0) / 1000.0)
        emitEvent(buildJsonObject {
            put("type", "session.usage.updated")
            put("usage", buildJsonObject { put("seconds", seconds) })
        })
        emitEvent(buildJsonObject {
            put("type", "session.closed")
            put("usage", buildJsonObject { put("seconds", seconds) })
        })
        disconnect()
    }

    override fun disconnect() {
        val expectedGeneration = synchronized(lock) {
            activeAttempt?.let(retiredAttempts::addLast)
            activeAttempt = null
            generation.incrementAndGet()
        }
        audioScope.launch {
            drainRetiredAttempts()
            emitZeroLevels(expectedGeneration)
        }
    }

    private fun listener(
        attempt: Attempt,
        resumed: Boolean,
        resumptionHandle: String?,
        socketGeneration: Long,
    ) = object : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            if (!isCurrent(attempt) || attempt.socketGeneration.get() != socketGeneration) {
                webSocket.cancel()
                return
            }
            attempt.socket = webSocket
            if (!sendNow(attempt, GeminiLiveProtocol.setup(attempt.instructions, attempt.language, resumptionHandle))) {
                fail(attempt, applicationContext.getString(R.string.error_transport_connection))
            }
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            if (!isCurrent(attempt) || attempt.socketGeneration.get() != socketGeneration) return
            if (text.toByteArray(Charsets.UTF_8).size !in 1..MAX_EVENT_BYTES) {
                fail(attempt, applicationContext.getString(R.string.error_transport_invalid_event))
                return
            }
            val message = try {
                JSON.parseToJsonElement(text).jsonObject
            } catch (_: Exception) {
                fail(attempt, applicationContext.getString(R.string.error_transport_invalid_event))
                return
            }
            handleMessage(attempt, message, resumed)
        }

        override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
            if (isCurrent(attempt) && attempt.socketGeneration.get() == socketGeneration) {
                fail(attempt, applicationContext.getString(R.string.error_transport_invalid_event))
            }
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            if (isCurrent(attempt) && attempt.socketGeneration.get() == socketGeneration && !attempt.closing.get()) {
                reconnectOrFail(attempt, resumed, applicationContext.getString(R.string.error_transport_channel_closed))
            }
        }

        override fun onFailure(webSocket: WebSocket, error: Throwable, response: Response?) {
            if (!isCurrent(attempt) || attempt.socketGeneration.get() != socketGeneration || attempt.closing.get()) return
            val message = response?.let(::httpFailureMessage)
                ?: applicationContext.getString(R.string.error_transport_network_lost)
            reconnectOrFail(attempt, resumed, message)
        }
    }

    private fun handleMessage(attempt: Attempt, message: JsonObject, resumed: Boolean) {
        if (!isCurrent(attempt)) return
        if (message["error"] is JsonObject) {
            val error = message["error"] as? JsonObject
            val code = (error?.get("code") as? JsonPrimitive)
                ?.contentOrNull?.toIntOrNull()
            fail(attempt, code?.let { httpFailureMessage(it, error) }
                ?: applicationContext.getString(R.string.error_transport_connection))
            return
        }
        val resumeUpdate = message["sessionResumptionUpdate"] as? JsonObject
        if (resumeUpdate != null) {
            val resumable = (resumeUpdate["resumable"] as? JsonPrimitive)?.contentOrNull == "true"
            val handle = resumeUpdate.string("newHandle")
            if (resumable && !handle.isNullOrBlank()) attempt.resumptionHandle = handle
        }
        if (message["goAway"] != null) {
            reconnect(attempt)
        }
        if (message["setupComplete"] != null) {
            if (resumed) {
                if (attempt.reconnecting.compareAndSet(true, false)) {
                    attempt.reconnectReady?.complete(Unit)
                }
            } else {
                if (!attempt.readyHandled.compareAndSet(false, true)) return
                val historyMessage = GeminiLiveProtocol.history(attempt.history)
                if (historyMessage != null && !sendNow(attempt, historyMessage)) {
                    fail(attempt, applicationContext.getString(R.string.error_transport_connection))
                    return
                }
                audioScope.launch {
                    if (!isCurrent(attempt)) return@launch
                    try {
                        startAudio(attempt)
                        attempt.startedAtMilliseconds = SystemClock.elapsedRealtime()
                        attempt.started.complete(Unit)
                        emitEvent(attempt, buildJsonObject { put("type", "session.started") })
                        startMetering(attempt)
                    } catch (_: Exception) {
                        fail(attempt, applicationContext.getString(R.string.error_transport_audio_stopped))
                    }
                }
            }
        }

        val content = message["serverContent"] as? JsonObject ?: return
        if ((content["interrupted"] as? JsonPrimitive)?.contentOrNull == "true") {
            interruptPlayback(attempt)
        }
        val modelTurn = content["modelTurn"] as? JsonObject
        val parts = modelTurn?.get("parts") as? JsonArray
        parts?.forEach { part ->
            val inline = (part as? JsonObject)?.get("inlineData") as? JsonObject ?: return@forEach
            val data = inline.string("data") ?: return@forEach
            val mimeType = inline.string("mimeType") ?: return@forEach
            if (mimeType != "audio/pcm;rate=24000") {
                fail(attempt, applicationContext.getString(R.string.error_transport_invalid_event))
                return@forEach
            }
            val pcm = try { java.util.Base64.getDecoder().decode(data) } catch (_: IllegalArgumentException) {
                fail(attempt, applicationContext.getString(R.string.error_transport_invalid_event))
                return
            }
            if (pcm.isEmpty() || pcm.size > MAX_AUDIO_BYTES || pcm.size % 2 != 0) {
                fail(attempt, applicationContext.getString(R.string.error_transport_invalid_event))
                return
            }
            playAudio(attempt, pcm)
        }

        (content["interimInputTranscription"] as? JsonObject)?.string("text")?.let {
            emitTranscript(attempt, SpeakerKind.INPUT, it)
        }
        (content["inputTranscription"] as? JsonObject)?.string("text")?.let {
            emitTranscript(attempt, SpeakerKind.INPUT, it)
            attempt.inputTranscript = ""
            attempt.inputTranscriptStartMilliseconds = null
        }
        (content["outputTranscription"] as? JsonObject)?.string("text")?.let {
            emitTranscript(attempt, SpeakerKind.OUTPUT, it)
        }
        if ((content["turnComplete"] as? JsonPrimitive)?.contentOrNull == "true") {
            attempt.inputTranscript = ""
            attempt.outputTranscript = ""
            attempt.inputTranscriptStartMilliseconds = null
            attempt.outputTranscriptStartMilliseconds = null
        }
    }

    private fun emitTranscript(attempt: Attempt, speaker: SpeakerKind, text: String) {
        if (text.length > MAX_TRANSCRIPT_CHARS) {
            fail(attempt, applicationContext.getString(R.string.error_transport_invalid_event))
            return
        }
        val previous = if (speaker == SpeakerKind.INPUT) attempt.inputTranscript else attempt.outputTranscript
        val (delta, updated) = GeminiLiveProtocol.transcriptDelta(previous, text)
        if (speaker == SpeakerKind.INPUT) attempt.inputTranscript = updated else attempt.outputTranscript = updated
        if (delta.isBlank()) return
        val now = (SystemClock.elapsedRealtime() - attempt.startedAtMilliseconds).toInt().coerceAtLeast(0)
        val start = if (speaker == SpeakerKind.INPUT) {
            attempt.inputTranscriptStartMilliseconds ?: now.also { attempt.inputTranscriptStartMilliseconds = it }
        } else {
            attempt.outputTranscriptStartMilliseconds ?: now.also { attempt.outputTranscriptStartMilliseconds = it }
        }
        emitEvent(attempt, buildJsonObject {
            put("type", if (speaker == SpeakerKind.INPUT) "session.input_transcript.delta" else "session.output_transcript.delta")
            put("delta", delta)
            put("start_ms", start)
            put("end_ms", maxOf(now, start))
            put("event_id", UUID.randomUUID().toString())
        })
    }

    private fun startAudio(attempt: Attempt) {
        val record = checkNotNull(attempt.audioRecord)
        val track = checkNotNull(attempt.audioTrack)
        record.startRecording()
        track.play()
        attempt.recordJob = ioScope.launch {
            val samples = ShortArray(INPUT_SAMPLES_PER_CHUNK)
            try {
                while (isActive && isCurrent(attempt)) {
                    val count = record.read(samples, 0, samples.size, AudioRecord.READ_BLOCKING)
                    if (count < 0) throw IOException("Microphone read failed")
                    if (count == 0) continue
                    val muted = attempt.muted.get()
                    val pcm = ByteArray(count * 2)
                    var sum = 0.0
                    for (index in 0 until count) {
                        val sample = if (muted) 0 else samples[index].toInt()
                        sum += sample.toDouble() * sample
                        pcm[index * 2] = sample.toByte()
                        pcm[index * 2 + 1] = (sample shr 8).toByte()
                    }
                    attempt.lastInputLevel = if (muted) 0.0 else normalizedLevel(sum, count)
                    emitLevels(attempt.lastInputLevel, attempt.lastOutputLevel, attempt)
                    sendAudio(attempt, pcm)
                }
            } catch (_: CancellationException) {
                throw
            } catch (_: Exception) {
                fail(attempt, applicationContext.getString(R.string.error_transport_audio_stopped))
            }
        }
    }

    private fun sendAudio(attempt: Attempt, pcm: ByteArray) {
        if (!isCurrent(attempt) || !attempt.started.isCompleted || attempt.reconnecting.get()) return
        if (attempt.socket?.queueSize()?.let { it > MAX_SOCKET_QUEUE_BYTES } == true) {
            fail(attempt, applicationContext.getString(R.string.error_transport_network_lost))
            return
        }
        val message = GeminiLiveProtocol.audioChunk(pcm)
        if (!sendNow(attempt, message)) {
            fail(attempt, applicationContext.getString(R.string.error_transport_channel_closed))
        }
    }

    private fun playAudio(attempt: Attempt, pcm: ByteArray) {
        val playbackGeneration = attempt.playbackGeneration.get()
        val samples = ShortArray(pcm.size / 2)
        var sum = 0.0
        for (index in samples.indices) {
            val value = ((pcm[index * 2 + 1].toInt() shl 8) or (pcm[index * 2].toInt() and 0xff)).toShort()
            samples[index] = value
            sum += value.toDouble() * value
        }
        attempt.lastOutputLevel = normalizedLevel(sum, samples.size)
        val outputLevel = attempt.lastOutputLevel
        emitLevels(attempt.lastInputLevel, outputLevel, attempt)
        audioScope.launch {
            if (!isCurrent(attempt) || attempt.playbackGeneration.get() != playbackGeneration) return@launch
            val track = attempt.audioTrack ?: return@launch
            var offset = 0
            while (offset < pcm.size && isCurrent(attempt) && attempt.playbackGeneration.get() == playbackGeneration) {
                val written = track.write(pcm, offset, pcm.size - offset, AudioTrack.WRITE_BLOCKING)
                if (written <= 0) {
                    fail(attempt, applicationContext.getString(R.string.error_transport_audio_stopped))
                    return@launch
                }
                offset += written
            }
            if (isCurrent(attempt) && attempt.playbackGeneration.get() == playbackGeneration) {
                attempt.lastOutputLevel = 0.0
                emitLevels(attempt.lastInputLevel, 0.0, attempt)
            }
        }
    }

    private fun interruptPlayback(attempt: Attempt) {
        attempt.playbackGeneration.incrementAndGet()
        audioScope.launch {
            if (!isCurrent(attempt)) return@launch
            try {
                attempt.audioTrack?.apply { pause(); flush(); play() }
                attempt.lastOutputLevel = 0.0
                emitLevels(attempt.lastInputLevel, 0.0, attempt)
            } catch (_: Exception) {
                fail(attempt, applicationContext.getString(R.string.error_transport_audio_stopped))
            }
        }
    }

    private fun startMetering(attempt: Attempt) {
        attempt.meterJob?.cancel()
        attempt.meterJob = audioScope.launch {
            while (isActive && isCurrent(attempt)) {
                emitLevels(if (attempt.muted.get()) 0.0 else attempt.lastInputLevel, attempt.lastOutputLevel, attempt)
                delay(METER_INTERVAL_MILLISECONDS)
            }
        }
    }

    private fun configureAudio(attempt: Attempt) {
        if (applicationContext.checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            throw GeminiLiveException(applicationContext.getString(R.string.error_transport_microphone))
        }
        if (attempt.previousAudioMode != AudioManager.MODE_NORMAL || audioManager.mode != AudioManager.MODE_NORMAL ||
            (Build.VERSION.SDK_INT < 31 && LegacyCommunicationAudioRoute.hasExistingSco(applicationContext, audioManager))) {
            throw GeminiLiveException(applicationContext.getString(R.string.error_transport_audio_focus))
        }
        val attributes = voiceAudioAttributes()
        val focusRequest = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT)
            .setAudioAttributes(attributes)
            .setAcceptsDelayedFocusGain(false)
            .setOnAudioFocusChangeListener({ change ->
                if (change == AudioManager.AUDIOFOCUS_LOSS ||
                    change == AudioManager.AUDIOFOCUS_LOSS_TRANSIENT ||
                    change == AudioManager.AUDIOFOCUS_LOSS_TRANSIENT_CAN_DUCK
                ) fail(attempt, applicationContext.getString(R.string.error_transport_audio_interrupted))
            }, Handler(Looper.getMainLooper()))
            .build()
        attempt.focusRequest = focusRequest
        if (audioManager.requestAudioFocus(focusRequest) != AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
            throw GeminiLiveException(applicationContext.getString(R.string.error_transport_audio_focus))
        }
        attempt.ownsAudioFocus = true
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
        attempt.ownsAudioMode = true
        if (Build.VERSION.SDK_INT < 31) {
            attempt.legacyAudioRoute = LegacyCommunicationAudioRoute(
                applicationContext, audioManager, audioScope, attempt.previousSpeakerphone,
                onFailure = { fail(attempt, applicationContext.getString(R.string.error_transport_audio_stopped)) },
            ).also { it.start() }
        }
        routeCommunicationAudio(attempt)
        val deviceCallback = object : AudioDeviceCallback() {
            override fun onAudioDevicesAdded(addedDevices: Array<out AudioDeviceInfo>) = reroute()
            override fun onAudioDevicesRemoved(removedDevices: Array<out AudioDeviceInfo>) = reroute()
            private fun reroute() {
                audioScope.launch {
                    if (isCurrent(attempt)) {
                        try { routeCommunicationAudio(attempt) }
                        catch (_: Exception) { fail(attempt, applicationContext.getString(R.string.error_transport_audio_stopped)) }
                    }
                }
            }
        }
        attempt.deviceCallback = deviceCallback
        audioManager.registerAudioDeviceCallback(deviceCallback, Handler(Looper.getMainLooper()))
    }

    private fun prepareAudio(attempt: Attempt) {
        val inputMinBuffer = AudioRecord.getMinBufferSize(
            INPUT_SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
        )
        if (inputMinBuffer <= 0) throw GeminiLiveException(applicationContext.getString(R.string.error_transport_audio_stopped))
        val record = AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.VOICE_COMMUNICATION)
            .setAudioFormat(AudioFormat.Builder()
                .setSampleRate(INPUT_SAMPLE_RATE)
                .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .build())
            .setBufferSizeInBytes(maxOf(inputMinBuffer * 2, INPUT_SAMPLES_PER_CHUNK * 4))
            .build()
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            record.release()
            throw GeminiLiveException(applicationContext.getString(R.string.error_transport_audio_stopped))
        }
        attempt.audioRecord = record
        attempt.echoCanceler = if (AcousticEchoCanceler.isAvailable()) AcousticEchoCanceler.create(record.audioSessionId) else null
        attempt.noiseSuppressor = if (NoiseSuppressor.isAvailable()) NoiseSuppressor.create(record.audioSessionId) else null
        attempt.automaticGainControl = if (AutomaticGainControl.isAvailable()) AutomaticGainControl.create(record.audioSessionId) else null
        attempt.echoCanceler?.enabled = true
        attempt.noiseSuppressor?.enabled = true
        attempt.automaticGainControl?.enabled = true

        val outputMinBuffer = AudioTrack.getMinBufferSize(
            OUTPUT_SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT,
        )
        if (outputMinBuffer <= 0) throw GeminiLiveException(applicationContext.getString(R.string.error_transport_audio_stopped))
        val track = AudioTrack.Builder()
            .setAudioAttributes(voiceAudioAttributes())
            .setAudioFormat(AudioFormat.Builder()
                .setSampleRate(OUTPUT_SAMPLE_RATE)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .build())
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(maxOf(outputMinBuffer, OUTPUT_SAMPLES_PER_CHUNK * 4))
            .build()
        if (track.state != AudioTrack.STATE_INITIALIZED) {
            track.release()
            throw GeminiLiveException(applicationContext.getString(R.string.error_transport_audio_stopped))
        }
        attempt.audioTrack = track
    }

    private fun routeCommunicationAudio(attempt: Attempt) {
        if (Build.VERSION.SDK_INT >= 31) {
            val current = audioManager.communicationDevice
            val selected = selectCommunicationDevice(
                current, audioManager.availableCommunicationDevices,
                sameDevice = { left, right -> left.id == right.id },
            ) { it.type }
            if (selected != null && current?.id != selected.id && audioManager.setCommunicationDevice(selected)) {
                attempt.ownsCommunicationRoute = true
            }
        } else attempt.legacyAudioRoute?.devicesChanged()
    }

    private fun sendNow(attempt: Attempt, message: JsonObject): Boolean {
        if (!isCurrent(attempt)) return false
        val socket = attempt.socket ?: return false
        return try { socket.send(message.toString()) } catch (_: Exception) { false }
    }

    private fun reconnect(attempt: Attempt) {
        if (!attempt.reconnecting.compareAndSet(false, true)) return
        val handle = attempt.resumptionHandle
        if (handle.isNullOrBlank()) {
            attempt.reconnecting.set(false)
            fail(attempt, applicationContext.getString(R.string.error_transport_channel_closed))
            return
        }
        audioScope.launch {
            val oldSocket = attempt.socket
            val ready = CompletableDeferred<Unit>()
            attempt.reconnectReady = ready
            if (oldSocket != null) attempt.retiredSockets += oldSocket
            try {
                attempt.socket = openSocket(attempt, resumed = true, resumptionHandle = handle)
                withTimeout(READY_TIMEOUT_MILLISECONDS) { ready.await() }
                attempt.reconnectReady = null
                oldSocket?.close(1000, null)
            } catch (_: Exception) {
                fail(attempt, applicationContext.getString(R.string.error_transport_network_lost))
            }
        }
    }

    private fun reconnectOrFail(attempt: Attempt, resumed: Boolean, message: String) {
        if (!resumed && !attempt.resumptionHandle.isNullOrBlank()) reconnect(attempt)
        else fail(attempt, message)
    }

    private fun request(key: String) = Request.Builder()
        .url(endpoint.newBuilder().addQueryParameter("key", key).build())
        .build()

    private fun openSocket(attempt: Attempt, resumed: Boolean, resumptionHandle: String?): WebSocket {
        val socketGeneration = attempt.socketGeneration.incrementAndGet()
        return client.newWebSocket(
            request(attempt.apiKey),
            listener(attempt, resumed, resumptionHandle, socketGeneration),
        )
    }

    private fun fail(attempt: Attempt, message: String) {
        if (!isCurrent(attempt) || attempt.closing.get() || !attempt.failureReported.compareAndSet(false, true)) return
        if (attempt.startedAtMilliseconds > 0) {
            val seconds = ((SystemClock.elapsedRealtime() - attempt.startedAtMilliseconds).coerceAtLeast(0) / 1000.0)
            emitEvent(buildJsonObject {
                put("type", "session.usage.updated")
                put("usage", buildJsonObject { put("seconds", seconds) })
            })
        }
        attempt.started.completeExceptionally(GeminiLiveException(message))
        audioScope.launch {
            if (!cleanupIfCurrent(attempt)) return@launch
            scope.launch { onFailure?.invoke(message) }
        }
    }

    private suspend fun cleanupIfCurrent(attempt: Attempt): Boolean = withContext(AUDIO_DISPATCHER) {
        synchronized(lock) {
            if (activeAttempt !== attempt) return@withContext false
            activeAttempt = null
            generation.incrementAndGet()
        }
        cleanup(attempt)
        true
    }

    private fun drainRetiredAttempts() {
        while (true) {
            val retired = synchronized(lock) { retiredAttempts.removeFirstOrNull() } ?: break
            cleanup(retired)
        }
    }

    private fun cleanup(attempt: Attempt) {
        if (!attempt.cleaned.compareAndSet(false, true)) return
        attempt.scopeCompletion?.dispose()
        attempt.meterJob?.cancel()
        attempt.recordJob?.cancel()
        attempt.reconnectReady?.cancel()
        attempt.started.cancel()
        try { attempt.socket?.cancel() } catch (_: Exception) { }
        attempt.retiredSockets.forEach { try { it.cancel() } catch (_: Exception) { } }
        try { attempt.audioRecord?.stop() } catch (_: Exception) { }
        try { attempt.audioTrack?.pause() } catch (_: Exception) { }
        try { attempt.audioTrack?.flush() } catch (_: Exception) { }
        try { attempt.echoCanceler?.release() } catch (_: Exception) { }
        try { attempt.noiseSuppressor?.release() } catch (_: Exception) { }
        try { attempt.automaticGainControl?.release() } catch (_: Exception) { }
        try { attempt.audioRecord?.release() } catch (_: Exception) { }
        try { attempt.audioTrack?.release() } catch (_: Exception) { }
        attempt.deviceCallback?.let { try { audioManager.unregisterAudioDeviceCallback(it) } catch (_: Exception) { } }
        attempt.legacyAudioRoute?.close(attempt.previousSpeakerphone)
        if (attempt.ownsCommunicationRoute && Build.VERSION.SDK_INT >= 31) {
            try { audioManager.clearCommunicationDevice() } catch (_: Exception) { }
        }
        if (attempt.ownsAudioMode) {
            try { audioManager.mode = AudioManager.MODE_NORMAL } catch (_: Exception) { }
        }
        if (attempt.ownsAudioFocus) {
            try { attempt.focusRequest?.let(audioManager::abandonAudioFocusRequest) } catch (_: Exception) { }
        }
    }

    private fun emitEvent(attempt: Attempt, event: JsonObject) {
        if (!isCurrent(attempt)) return
        scope.launch { if (isCurrent(attempt)) onEvent?.invoke(event) }
    }

    private fun emitEvent(event: JsonObject) { scope.launch { onEvent?.invoke(event) } }

    private fun emitLevels(input: Double, output: Double, attempt: Attempt? = null) {
        scope.launch {
            if (attempt == null || isCurrent(attempt)) onLevels?.invoke(input, output)
        }
    }

    private fun emitZeroLevels(expectedGeneration: Long) {
        scope.launch {
            if (generation.get() == expectedGeneration) onLevels?.invoke(0.0, 0.0)
        }
    }

    private fun isCurrent(attempt: Attempt): Boolean =
        activeAttempt === attempt && generation.get() == attempt.id && !attempt.cleaned.get()

    private fun requireCurrent(attempt: Attempt) {
        if (!isCurrent(attempt)) throw CancellationException("Gemini voice connection superseded")
    }

    private fun voiceAudioAttributes() = AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build()

    private fun httpFailureMessage(response: Response): String {
        val error = runCatching {
            JSON.parseToJsonElement(response.peekBody(MAX_ERROR_BYTES).string()).jsonObject["error"] as? JsonObject
        }.getOrNull()
        return httpFailureMessage(response.code, error)
    }

    private fun httpFailureMessage(status: Int, error: JsonObject? = null): String {
        val reason = (error?.get("details") as? JsonArray)?.firstNotNullOfOrNull {
            ((it as? JsonObject)?.get("reason") as? JsonPrimitive)?.contentOrNull
        }
        return when {
            status == 400 && reason == "API_KEY_INVALID" -> applicationContext.getString(R.string.error_gemini_401)
            status == 401 -> applicationContext.getString(R.string.error_gemini_401)
            status == 403 || status == 404 -> applicationContext.getString(R.string.error_gemini_403_404)
            status == 429 -> applicationContext.getString(R.string.error_gemini_429)
            status == 400 || status == 422 -> applicationContext.getString(R.string.error_provider_request)
            status == 408 || status >= 500 -> applicationContext.getString(R.string.error_provider_unavailable)
            else -> applicationContext.getString(R.string.error_gemini_generic, status)
        }
    }

    private fun normalizedLevel(sum: Double, count: Int): Double =
        if (count == 0) 0.0 else minOf(1.0, sqrt(sum / count) / Short.MAX_VALUE * 4.0)

    private enum class SpeakerKind { INPUT, OUTPUT }

    private class Attempt(
        val id: Long,
        val previousAudioMode: Int,
        val previousSpeakerphone: Boolean,
        val apiKey: String,
        val instructions: String,
        val history: JsonArray,
        val language: String?,
    ) {
        var socket: WebSocket? = null
        var audioRecord: AudioRecord? = null
        var audioTrack: AudioTrack? = null
        var echoCanceler: AcousticEchoCanceler? = null
        var noiseSuppressor: NoiseSuppressor? = null
        var automaticGainControl: AutomaticGainControl? = null
        var focusRequest: AudioFocusRequest? = null
        var ownsAudioFocus = false
        var ownsAudioMode = false
        var ownsCommunicationRoute = false
        var legacyAudioRoute: LegacyCommunicationAudioRoute? = null
        var deviceCallback: AudioDeviceCallback? = null
        var recordJob: Job? = null
        var meterJob: Job? = null
        var scopeCompletion: kotlinx.coroutines.DisposableHandle? = null
        var startedAtMilliseconds = 0L
        var resumptionHandle: String? = null
        var reconnectReady: CompletableDeferred<Unit>? = null
        val retiredSockets = mutableListOf<WebSocket>()
        val socketGeneration = AtomicLong(0)
        var inputTranscript = ""
        var outputTranscript = ""
        var inputTranscriptStartMilliseconds: Int? = null
        var outputTranscriptStartMilliseconds: Int? = null
        @Volatile var lastInputLevel = 0.0
        @Volatile var lastOutputLevel = 0.0
        val started = CompletableDeferred<Unit>()
        val muted = AtomicBoolean(false)
        val readyHandled = AtomicBoolean(false)
        val reconnecting = AtomicBoolean(false)
        val closing = AtomicBoolean(false)
        val failureReported = AtomicBoolean(false)
        val cleaned = AtomicBoolean(false)
        val playbackGeneration = AtomicLong(0)
    }

    private fun JsonObject.string(key: String): String? =
        (this[key] as? JsonPrimitive)?.takeIf { it.isString }?.contentOrNull

    companion object {
        private const val INPUT_SAMPLE_RATE = 16_000
        private const val OUTPUT_SAMPLE_RATE = 24_000
        private const val INPUT_SAMPLES_PER_CHUNK = 1_600
        private const val OUTPUT_SAMPLES_PER_CHUNK = 2_400
        private const val READY_TIMEOUT_MILLISECONDS = 20_000L
        private const val METER_INTERVAL_MILLISECONDS = 100L
        private const val MAX_EVENT_BYTES = 2_097_152
        private const val MAX_AUDIO_BYTES = 262_144
        private const val MAX_TRANSCRIPT_CHARS = 50_000
        private const val MAX_SOCKET_QUEUE_BYTES = 1_048_576L
        private const val MAX_ERROR_BYTES = 16_384L
        private val LIVE_ENDPOINT = HttpUrl.Builder()
            .scheme("https")
            .host("generativelanguage.googleapis.com")
            .addPathSegments("ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")
            .build()
        private val JSON = Json { ignoreUnknownKeys = true }
        private val AUDIO_DISPATCHER = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "mural-gemini-audio-control").apply { isDaemon = true }
        }.asCoroutineDispatcher()

        private fun defaultClient() = OkHttpClient.Builder()
            .connectTimeout(20, java.util.concurrent.TimeUnit.SECONDS)
            .readTimeout(0, java.util.concurrent.TimeUnit.MILLISECONDS)
            .writeTimeout(20, java.util.concurrent.TimeUnit.SECONDS)
            .followRedirects(false)
            .followSslRedirects(false)
            .cookieJar(CookieJar.NO_COOKIES)
            .cache(null)
            .build()
    }

}

internal class GeminiLiveException(message: String) : IOException(message)
