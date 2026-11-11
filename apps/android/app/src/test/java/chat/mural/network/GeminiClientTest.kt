package chat.mural.network

import chat.mural.core.ProviderFailureKind
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.*
import okhttp3.OkHttpClient
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class GeminiClientTest {
    private lateinit var server: MockWebServer
    private lateinit var gemini: GeminiClient

    @Before fun setup() {
        server = MockWebServer(); server.start()
        gemini = GeminiClient("AIza-fake-test-key-only-0000", OkHttpClient.Builder().followRedirects(false).build(), server.url("/v1beta/"))
    }
    @After fun teardown() { server.shutdown() }

    private fun reply(text: String = "Hola", extra: String = "", finish: String = "STOP") =
        """{"candidates":[{"content":{"parts":[{"text":"$text"}]},"finishReason":"$finish"$extra}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":5,"thoughtsTokenCount":2}}"""

    @Test fun sendsTheKeyInAHeaderAndNeverInTheUrl() = runBlocking {
        server.enqueue(MockResponse().setBody(reply()))
        val result = gemini.respond("policy", "Hei")
        assertEquals("Hola", result.text); assertEquals(APIUsage(12, 7, 0), result.usage)
        val sent = server.takeRequest()
        assertEquals("/v1beta/models/${GeminiClient.MODEL}:generateContent", sent.path)
        assertEquals("AIza-fake-test-key-only-0000", sent.getHeader("x-goog-api-key"))
        assertNull(sent.getHeader("Authorization"))
        val body = Json.parseToJsonElement(sent.body.readUtf8()).jsonObject
        assertEquals("policy", body["systemInstruction"]!!.jsonObject["parts"]!!.jsonArray[0].jsonObject["text"]!!.jsonPrimitive.content)
        assertEquals("Hei", body["contents"]!!.jsonArray[0].jsonObject["parts"]!!.jsonArray[0].jsonObject["text"]!!.jsonPrimitive.content)
        assertNull(body["tools"]); assertNull(body["generationConfig"]!!.jsonObject["responseJsonSchema"])
    }

    @Test fun structuredRequestsCarryTheSchemaAndSearchRequestsEnableGrounding() = runBlocking {
        server.enqueue(MockResponse().setBody(reply("{}")))
        server.enqueue(MockResponse().setBody(reply()))
        val schema = buildJsonObject { put("type", "object") }
        gemini.respond("policy", "x", schema = schema)
        gemini.respond("policy", "x", search = true)
        val structured = Json.parseToJsonElement(server.takeRequest().body.readUtf8()).jsonObject["generationConfig"]!!.jsonObject
        assertEquals(schema, structured["responseJsonSchema"]); assertEquals("application/json", structured["responseMimeType"]!!.jsonPrimitive.content)
        val searched = Json.parseToJsonElement(server.takeRequest().body.readUtf8()).jsonObject
        assertNotNull(searched["tools"]!!.jsonArray[0].jsonObject["googleSearch"])
    }

    @Test fun keepsOnlySafeHttpsGroundingSourcesAndSkipsReasoningParts() = runBlocking {
        val grounding = ""","groundingMetadata":{"webSearchQueries":["q"],"groundingChunks":[
            {"web":{"uri":"https://example.com/a","title":"Example"}},{"web":{"uri":"http://insecure.test/b","title":"Plain"}},
            {"web":{"uri":"https://user:pw@example.com/c","title":"Credentials"}},{"web":{"uri":"https://example.com/a","title":"Duplicate"}}]}"""
        val body = """{"candidates":[{"content":{"parts":[{"text":"private reasoning","thought":true},{"text":"Answer"}]},"finishReason":"STOP"$grounding}]}"""
        server.enqueue(MockResponse().setBody(body))
        val result = gemini.respond("policy", "x", search = true)
        assertEquals("Answer", result.text)
        assertEquals(listOf("https://example.com/a"), result.sources.map { it.url }); assertEquals(1, result.usage.searches)
    }

    @Test fun missingKeyNeverReachesTheNetwork() = runBlocking {
        val none = GeminiClient(null, OkHttpClient(), server.url("/v1beta/"))
        try { none.respond("policy", "x"); fail("sent without a key") } catch (_: GeminiException.MissingKey) { }
        assertEquals(0, server.requestCount)
    }

    @Test fun errorsAreClassifiedWithoutExposingProviderMessages() = runBlocking {
        val badKey = """{"error":{"code":400,"message":"private text","status":"INVALID_ARGUMENT","details":[{"reason":"API_KEY_INVALID"}]}}"""
        val cases = listOf(400 to badKey to ProviderFailureKind.authentication, 403 to "{}" to ProviderFailureKind.modelAccess,
            429 to "not json" to ProviderFailureKind.rateLimit, 503 to "" to ProviderFailureKind.unavailable,
            400 to """{"error":{"status":"INVALID_ARGUMENT"}}""" to ProviderFailureKind.invalidRequest)
        for ((statusAndBody, kind) in cases) {
            server.enqueue(MockResponse().setResponseCode(statusAndBody.first).setBody(statusAndBody.second))
            try { gemini.respond("policy", "x"); fail("accepted an error") }
            catch (error: GeminiException.Http) {
                assertEquals(kind, error.kind); assertEquals(statusAndBody.first, error.status)
                assertFalse(error.message.orEmpty().contains("private"))
            }
        }
        assertEquals(cases.size, server.requestCount)
    }

    @Test fun blockedTruncatedAndEmptyRepliesNeverBecomeSuccessfulAnswers() = runBlocking {
        val bodies = listOf(
            """{"promptFeedback":{"blockReason":"SAFETY"}}""" to GeminiException.Refused,
            reply(finish = "SAFETY") to GeminiException.Refused,
            reply(finish = "MAX_TOKENS") to GeminiException.Incomplete,
            """{"candidates":[]}""" to GeminiException.Incomplete,
            reply(text = " ") to GeminiException.Incomplete,
            "not json" to GeminiException.InvalidResponse,
        )
        for ((body, expected) in bodies) {
            server.enqueue(MockResponse().setBody(body))
            try { gemini.respond("policy", "x"); fail("accepted $body") } catch (error: GeminiException) { assertSame(expected, error) }
        }
    }

    @Test fun keyShapesAreCheckedWithoutRejectingNewerGoogleFormats() {
        assertTrue(isGeminiKey("AIza" + "x".repeat(35))); assertTrue(isGeminiKey("AQ.Ab8-" + "y".repeat(30)))
        assertFalse(isGeminiKey("short")); assertFalse(isGeminiKey("has space " + "x".repeat(30))); assertFalse(isGeminiKey("é".repeat(30)))
        assertTrue(isOpenAIKey("sk-" + "x".repeat(30))); assertFalse(isOpenAIKey("AIza" + "x".repeat(35)))
        // Each provider refuses the other's key, so a key cannot be saved under the wrong provider by accident.
        assertFalse(isGeminiKey("sk-" + "x".repeat(30)))
    }

    @Test fun streamMeaningDeliversTheFullTextInOneCallback() = runBlocking {
        server.enqueue(MockResponse().setBody(reply("meaning text")))
        val chunks = mutableListOf<String>()
        val result = gemini.streamMeaning("policy", "input") { chunks += it }
        assertEquals("meaning text", result.text)
        assertEquals(listOf("meaning text"), chunks)
        assertEquals(1, server.requestCount)
    }
}
