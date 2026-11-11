package chat.mural.network

import chat.mural.core.SourceLink
import java.net.URI
import kotlinx.serialization.json.*

/** Decodes a Gemini generateContent reply and keeps only safe https citations from Google Search grounding. */
internal fun decodeGeminiResponse(response: JsonObject): APIResult {
    if (response.obj("promptFeedback")?.string("blockReason") != null) throw GeminiException.Refused
    val candidate = response.array("candidates").firstOrNull() as? JsonObject ?: throw GeminiException.Incomplete
    when (candidate.string("finishReason")) {
        null, "STOP" -> Unit
        "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY" -> throw GeminiException.Refused
        else -> throw GeminiException.Incomplete // MAX_TOKENS, RECITATION and anything unknown
    }

    val text = StringBuilder()
    for (partElement in candidate.obj("content")?.array("parts").orEmpty()) {
        val part = partElement as? JsonObject ?: continue
        if ((part["thought"] as? JsonPrimitive)?.booleanOrNull == true) continue // reasoning summaries are not replies
        part.string("text")?.let(text::append)
    }
    if (text.isBlank()) throw GeminiException.Incomplete

    val metadata = candidate.obj("groundingMetadata")
    val sources = linkedMapOf<String, SourceLink>()
    for (chunk in metadata?.array("groundingChunks").orEmpty()) {
        val web = (chunk as? JsonObject)?.obj("web") ?: continue
        val url = web.string("uri") ?: continue
        if (isSafeSourceUrl(url)) sources.putIfAbsent(url, SourceLink(web.string("title") ?: "Source", url))
    }
    val usage = response.obj("usageMetadata")
    return APIResult(
        text = text.toString(),
        sources = sources.values.toList(),
        usage = APIUsage(
            input = usage.tokens("promptTokenCount"),
            output = (usage.tokens("candidatesTokenCount") + usage.tokens("thoughtsTokenCount")).coerceAtMost(1_000_000_000),
            searches = metadata?.array("webSearchQueries")?.size ?: 0,
        ),
    )
}

// Same rule as TeachingResponse.kt, kept private here so the OpenAI decoder stays untouched.
private fun isSafeSourceUrl(value: String): Boolean = try {
    val uri = URI(value)
    uri.scheme == "https" && !uri.host.isNullOrBlank() && uri.userInfo == null
} catch (_: Exception) {
    false
}

private fun JsonObject.string(key: String): String? = (this[key] as? JsonPrimitive)?.contentOrNull
private fun JsonObject.obj(key: String): JsonObject? = this[key] as? JsonObject
private fun JsonObject.array(key: String): JsonArray = this[key] as? JsonArray ?: JsonArray(emptyList())
private fun JsonObject?.tokens(key: String): Int =
    ((this?.get(key) as? JsonPrimitive)?.intOrNull ?: 0).coerceIn(0, 1_000_000_000)
