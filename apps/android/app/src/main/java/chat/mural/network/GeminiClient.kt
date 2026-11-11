package chat.mural.network

import chat.mural.core.ProviderFailureKind
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import okhttp3.Call
import okhttp3.Callback
import okhttp3.CookieJar
import okhttp3.HttpUrl
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okio.Buffer

/**
 * Text-only Gemini adapter for the learner's own key. It answers the same helper requests as [APIClient]
 * (written replies, meanings, lookups, topics and assessments). Voice conversations stay on OpenAI.
 * The key travels in a header, never in the URL, and no request or response body is logged.
 */
class GeminiClient private constructor(
    private val readCredential: () -> String?,
    private val client: OkHttpClient = defaultClient(),
    private val baseUrl: HttpUrl = API_BASE_URL,
) : TeachingClient {
    constructor(credentials: CredentialStore) : this(credentials::read)

    internal constructor(key: String?, client: OkHttpClient, baseUrl: HttpUrl) :
        this({ key }, client, baseUrl)

    override suspend fun respond(
        instructions: String,
        input: String,
        schema: JsonObject?,
        search: Boolean,
        purpose: HelperPurpose?,
    ): APIResult = decodeGeminiResponse(post(requestBody(instructions, input, schema, search)))

    /**
     * Gemini does not expose the same SSE event stream as OpenAI Responses.
     * Deliver the complete meaning in one callback so the UI contract is preserved
     * without changing the OpenAI streaming path.
     */
    override suspend fun streamMeaning(
        instructions: String,
        input: String,
        onText: (String) -> Unit,
    ): APIResult {
        val result = respond(instructions, input, schema = null, search = false, purpose = HelperPurpose.MEANING)
        onText(result.text)
        return result
    }

    private fun requestBody(instructions: String, input: String, schema: JsonObject?, search: Boolean): JsonObject =
        buildJsonObject {
            put("systemInstruction", buildJsonObject {
                put("parts", buildJsonArray { add(buildJsonObject { put("text", instructions) }) })
            })
            put("contents", buildJsonArray {
                add(buildJsonObject {
                    put("role", "user")
                    put("parts", buildJsonArray { add(buildJsonObject { put("text", input) }) })
                })
            })
            put("generationConfig", buildJsonObject {
                // Thinking tokens count against this limit on current models, so it is larger than OpenAI's.
                put("maxOutputTokens", if (schema == null) TEXT_OUTPUT_TOKENS else STRUCTURED_OUTPUT_TOKENS)
                if (schema != null) {
                    put("responseMimeType", "application/json")
                    put("responseJsonSchema", schema)
                }
            })
            // Mural never asks for a schema and search in the same request.
            if (search) put("tools", buildJsonArray { add(buildJsonObject { put("googleSearch", buildJsonObject { }) }) })
        }

    private suspend fun post(body: JsonObject): JsonObject {
        val key = readCredential() ?: throw GeminiException.MissingKey
        val request = Request.Builder()
            .url(baseUrl.newBuilder().addPathSegments("models/$MODEL:generateContent").build())
            .header("x-goog-api-key", key)
            .header("Content-Type", JSON_MEDIA_TYPE.toString())
            .post(body.toString().toRequestBody(JSON_MEDIA_TYPE))
            .build()

        // Parse on OkHttp's worker while the continuation stays cancellable, as APIClient does.
        return suspendCancellableCoroutine { continuation ->
            val call = client.newCall(request)
            continuation.invokeOnCancellation { call.cancel() }
            call.enqueue(object : Callback {
                override fun onFailure(call: Call, error: IOException) {
                    if (continuation.isActive) continuation.resumeWithException(error)
                }
                override fun onResponse(call: Call, response: Response) {
                    try {
                        val value = response.use {
                            if (it.code !in 200..299) {
                                val reason = runCatching {
                                    errorReason(JSON.parseToJsonElement(it.peekBody(16_384).string()).jsonObject)
                                }.getOrNull()
                                throw GeminiException.Http(it.code, reason)
                            }
                            val payload = it.readBoundedBody()
                            try { JSON.parseToJsonElement(payload).jsonObject }
                            catch (_: Exception) { throw GeminiException.InvalidResponse }
                        }
                        if (continuation.isActive) continuation.resume(value)
                    } catch (error: Exception) {
                        if (continuation.isActive) continuation.resumeWithException(error)
                    }
                }
            })
        }
    }

    /** Gemini reports a bad key as HTTP 400, so the machine-readable reason is needed to tell it from a bad request. */
    private fun errorReason(payload: JsonObject): String? {
        val error = payload["error"] as? JsonObject ?: return null
        val details = error["details"] as? JsonArray ?: return null
        return details.firstNotNullOfOrNull { ((it as? JsonObject)?.get("reason") as? JsonPrimitive)?.contentOrNull }
    }

    private fun Response.readBoundedBody(): String {
        val responseBody = body ?: throw GeminiException.InvalidResponse
        if (responseBody.contentLength() > MAX_RESPONSE_BYTES) throw GeminiException.InvalidResponse
        val source = responseBody.source()
        val buffer = Buffer()
        var total = 0L
        while (true) {
            val count = source.read(buffer, minOf(8_192L, MAX_RESPONSE_BYTES + 1L - total))
            if (count == -1L) break
            total += count
            if (total > MAX_RESPONSE_BYTES) throw GeminiException.InvalidResponse
        }
        return buffer.readString(Charsets.UTF_8)
    }

    companion object {
        /** Taken from Google's current Gemini API examples. Change this one value to use another model. */
        internal const val MODEL = "gemini-3.8-flash"
        private const val TEXT_OUTPUT_TOKENS = 4_096
        private const val STRUCTURED_OUTPUT_TOKENS = 8_192
        private val API_BASE_URL = HttpUrl.Builder()
            .scheme("https")
            .host("generativelanguage.googleapis.com")
            .addPathSegment("v1beta")
            .addPathSegment("")
            .build()
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
        private const val MAX_RESPONSE_BYTES = 1_048_576L
        private val JSON = Json { ignoreUnknownKeys = true }

        private fun defaultClient() = OkHttpClient.Builder()
            .connectTimeout(45, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(45, TimeUnit.SECONDS)
            .callTimeout(60, TimeUnit.SECONDS)
            .followRedirects(false)
            .followSslRedirects(false)
            .cookieJar(CookieJar.NO_COOKIES)
            .cache(null)
            .build()
    }
}

/** Failures from the Gemini adapter. Messages are fixed text; provider error bodies are never surfaced. */
sealed class GeminiException(message: String) : IOException(message) {
    data object MissingKey : GeminiException("Add your Gemini key in Settings to begin.")
    data object InvalidResponse : GeminiException("Gemini returned an incomplete response. Please try again.")
    data object Incomplete : GeminiException("Gemini returned an incomplete response. Please try again.")
    data object Refused : GeminiException("Mural couldn't complete that request. Try a different topic.")
    class Http(val status: Int, reason: String? = null) : GeminiException("Gemini couldn't complete the request (HTTP $status).") {
        val kind: ProviderFailureKind = when {
            status == 401 || reason == "API_KEY_INVALID" -> ProviderFailureKind.authentication
            status == 403 || status == 404 -> ProviderFailureKind.modelAccess
            status == 429 -> ProviderFailureKind.rateLimit
            status == 408 || status >= 500 -> ProviderFailureKind.unavailable
            status == 400 || status == 422 -> ProviderFailureKind.invalidRequest
            else -> ProviderFailureKind.unknown
        }
    }
}
