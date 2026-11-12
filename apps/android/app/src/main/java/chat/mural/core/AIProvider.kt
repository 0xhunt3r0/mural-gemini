package chat.mural.core

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Which AI service handles voice and learning requests when the learner uses a personal key.
 * Each provider keeps its own key, so switching never removes either one.
 */
enum class AIProvider { OPENAI, GEMINI }

/** Holds only the provider name. It never contains credentials and is excluded from backups with the other preferences. */
internal class AIProviderStore(context: Context) {
    private val preferences = context.getSharedPreferences("mural_ai_provider", Context.MODE_PRIVATE)
    suspend fun read(): AIProvider = withContext(Dispatchers.IO) {
        runCatching { AIProvider.valueOf(preferences.getString("selection", null).orEmpty()) }
            .getOrDefault(AIProvider.OPENAI)
    }
    suspend fun select(provider: AIProvider) = withContext(Dispatchers.IO) {
        check(preferences.edit().putString("selection", provider.name).commit())
    }
}
