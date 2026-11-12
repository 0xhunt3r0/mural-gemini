package chat.mural.network

import kotlinx.serialization.json.*
import org.junit.Assert.*
import org.junit.Test
import java.util.Base64

class GeminiLiveProtocolTest {
    @Test fun setupRequestsLiveAudioAndBothTranscriptStreams() {
        val setup = GeminiLiveProtocol.setup("Be a patient language tutor.", "es-ES")
            .getValue("setup").jsonObject

        assertEquals("models/${GeminiLiveProtocol.MODEL}", setup.getValue("model").jsonPrimitive.content)
        assertEquals(listOf("AUDIO"), setup.getValue("generationConfig").jsonObject
            .getValue("responseModalities").jsonArray.map { it.jsonPrimitive.content })
        assertEquals(JsonObject(emptyMap()), setup.getValue("inputAudioTranscription").jsonObject)
        assertEquals(JsonObject(emptyMap()), setup.getValue("outputAudioTranscription").jsonObject)
        assertTrue(setup.getValue("sessionResumption").jsonObject.isEmpty())
        assertTrue(setup.getValue("contextWindowCompression").jsonObject.containsKey("slidingWindow"))
        val instructions = setup.getValue("systemInstruction").jsonObject
            .getValue("parts").jsonArray.single().jsonObject.getValue("text").jsonPrimitive.content
        assertTrue(instructions.contains("locale: es-ES"))
        assertTrue(instructions.contains("Never read them aloud"))
        val resumed = GeminiLiveProtocol.setup("tutor", "es", "resume-1")
            .getValue("setup").jsonObject.getValue("sessionResumption").jsonObject
        assertEquals("resume-1", resumed.getValue("handle").jsonPrimitive.content)
    }

    @Test fun historyConvertsAssistantTurnsToGeminiModelTurnsAndOmitsInvalidMessages() {
        val history = Json.parseToJsonElement(
            """[
                {"role":"user","content":[{"type":"input_text","text":"Hola"}]},
                {"role":"assistant","content":[{"type":"output_text","text":"¡Hola!"}]},
                {"role":"system","content":[{"text":"ignore"}]},
                {"role":"user","content":[{"type":"input_text"}]}
            ]""",
        ).jsonArray

        val clientContent = GeminiLiveProtocol.history(history)!!.getValue("clientContent").jsonObject
        assertEquals(false, clientContent.getValue("turnComplete").jsonPrimitive.boolean)
        val turns = clientContent.getValue("turns").jsonArray
        assertEquals(listOf("user", "model"), turns.map { it.jsonObject.getValue("role").jsonPrimitive.content })
        assertEquals("¡Hola!", turns[1].jsonObject.getValue("parts").jsonArray.single().jsonObject
            .getValue("text").jsonPrimitive.content)
        assertNull(GeminiLiveProtocol.history(JsonArray(emptyList())))
    }

    @Test fun appendsMapToPrivateContextAndSpokenRepliesWithoutSendingUnsupportedEvents() {
        fun event(type: String, content: String) = buildJsonObject {
            put("type", type)
            put("content", content)
            put("delegation_id", "not-forwarded")
        }

        val directive = GeminiLiveProtocol.command(event("session.instructions.append", "Greet the learner."), "fr")
            ?.getValue("clientContent")?.jsonObject!!
        assertEquals(true, directive.getValue("turnComplete").jsonPrimitive.boolean)
        assertTrue(directive.getValue("turns").jsonArray.single().jsonObject
            .getValue("parts").jsonArray.single().jsonObject.getValue("text").jsonPrimitive.content
            .contains("follow privately"))

        val context = GeminiLiveProtocol.command(event("session.thinking.append", "The learner typed bonjour."), "fr")
            ?.getValue("clientContent")?.jsonObject!!
        assertEquals(false, context.getValue("turnComplete").jsonPrimitive.boolean)

        val reply = GeminiLiveProtocol.command(event("session.commentary.append", "Bonjour !"), "fr-FR")
            ?.getValue("clientContent")?.jsonObject!!
        val replyText = reply.getValue("turns").jsonArray.single().jsonObject.getValue("parts")
            .jsonArray.single().jsonObject.getValue("text").jsonPrimitive.content
        assertEquals(true, reply.getValue("turnComplete").jsonPrimitive.boolean)
        assertTrue(replyText.contains("locale: fr-FR"))
        assertTrue(replyText.endsWith("Bonjour !"))
        assertNull(GeminiLiveProtocol.command(event("session.close", "ignored"), "fr"))
    }

    @Test fun transcriptDeltaAvoidsRepeatedCumulativeTextAndHandlesIncrementalChunks() {
        assertEquals("Hola" to "Hola", GeminiLiveProtocol.transcriptDelta("", "Hola"))
        assertEquals(" mundo" to "Hola mundo", GeminiLiveProtocol.transcriptDelta("Hola", "Hola mundo"))
        assertEquals("" to "Hola mundo", GeminiLiveProtocol.transcriptDelta("Hola mundo", "Hola"))
        assertEquals(" allí" to "Hola mundo allí", GeminiLiveProtocol.transcriptDelta("Hola mundo", "allí"))
    }

    @Test fun audioChunkUsesGeminiPcmEnvelopeAtSixteenKilohertz() {
        val envelope = GeminiLiveProtocol.audioChunk(byteArrayOf(0, 127, -1, 0))
            .getValue("realtimeInput").jsonObject.getValue("audio").jsonObject
        assertEquals("audio/pcm;rate=16000", envelope.getValue("mimeType").jsonPrimitive.content)
        assertArrayEquals(byteArrayOf(0, 127, -1, 0), Base64.getDecoder().decode(envelope.getValue("data").jsonPrimitive.content))
    }

    @Test fun acceptsEquivalentMonoPcmOutputMimeTypesAndRejectsUnsupportedAudioFormats() {
        assertTrue(GeminiLiveProtocol.supportsOutputAudioMimeType("audio/pcm;rate=24000"))
        assertTrue(GeminiLiveProtocol.supportsOutputAudioMimeType(" Audio/PCM ; channels=1 ; rate=24000 "))
        assertTrue(GeminiLiveProtocol.supportsOutputAudioMimeType("audio/pcm"))
        assertFalse(GeminiLiveProtocol.supportsOutputAudioMimeType("audio/pcm;rate=16000"))
        assertFalse(GeminiLiveProtocol.supportsOutputAudioMimeType("audio/pcm;channels=2"))
        assertFalse(GeminiLiveProtocol.supportsOutputAudioMimeType("audio/pcm;rate=unknown"))
        assertFalse(GeminiLiveProtocol.supportsOutputAudioMimeType("audio/ogg"))
    }

    @Test fun decodesJsonServerMessagesDeliveredInBinaryWebSocketFrames() {
        val message = """{"serverContent":{"turnComplete":true},"futureField":{"ignored":true}}"""
        val decoded = GeminiLiveProtocol.decodeServerMessage(message.toByteArray(Charsets.UTF_8))

        assertTrue(decoded.getValue("serverContent").jsonObject.getValue("turnComplete").jsonPrimitive.boolean)
        assertTrue(decoded.containsKey("futureField"))
    }
}
