package codes.felix.whisp.ui

import android.content.Intent
import android.net.Uri
import android.widget.Toast
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import codes.felix.whisp.audio.OpusDecoder
import codes.felix.whisp.whisper.WhisperEngine
import codes.felix.whisp.whisper.WhisperResult
import kotlinx.coroutines.delay

sealed class TranscribeState {
    object Loading : TranscribeState()
    data class Streaming(val textSoFar: String) : TranscribeState()
    data class Done(val result: WhisperResult) : TranscribeState()
    data class Error(val message: String) : TranscribeState()
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TranscriptScreen(
    audioUri: Uri,
    engine: WhisperEngine,
    onFinish: () -> Unit
) {
    var state by remember { mutableStateOf<TranscribeState>(TranscribeState.Loading) }
    var elapsedSeconds by remember { mutableIntStateOf(0) }
    var retryTrigger by remember { mutableIntStateOf(0) }
    val context = LocalContext.current
    val clipboard = LocalClipboardManager.current
    val scrollState = rememberScrollState()

    // Elapsed time counter
    LaunchedEffect(state) {
        if (state is TranscribeState.Loading || state is TranscribeState.Streaming) {
            elapsedSeconds = 0
            while (true) {
                delay(1000)
                elapsedSeconds++
            }
        }
    }

    // Auto-scroll when new text arrives
    LaunchedEffect(state) {
        if (state is TranscribeState.Streaming) {
            scrollState.animateScrollTo(scrollState.maxValue)
        }
    }

    // Transcription effect
    LaunchedEffect(retryTrigger) {
        state = TranscribeState.Loading
        try {
            val pcm = OpusDecoder.decodeToFloatArray(context, audioUri)
            state = TranscribeState.Streaming("")
            var accumulated = ""
            engine.transcribeStreaming(pcm).collect { event ->
                when (event) {
                    is WhisperEngine.StreamEvent.Segment -> {
                        accumulated += event.text
                        state = TranscribeState.Streaming(accumulated.trim())
                    }
                    is WhisperEngine.StreamEvent.Done -> {
                        state = TranscribeState.Done(event.result)
                    }
                }
            }
        } catch (e: Exception) {
            state = TranscribeState.Error(e.message ?: "Unknown error")
        }
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        when (val s = state) {
            is TranscribeState.Loading -> {
                Column(
                    modifier = Modifier.fillMaxSize(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    CircularProgressIndicator()
                    Spacer(modifier = Modifier.height(16.dp))
                    Text("Decoding audio...", style = MaterialTheme.typography.bodyLarge)
                }
            }

            is TranscribeState.Streaming -> {
                Column(modifier = Modifier.fillMaxSize()) {
                    TopAppBar(title = { Text("Transcribing...") })

                    SelectionContainer(
                        modifier = Modifier
                            .weight(1f)
                            .padding(16.dp)
                            .verticalScroll(scrollState)
                    ) {
                        Text(
                            text = s.textSoFar,
                            style = MaterialTheme.typography.bodyLarge,
                            lineHeight = 26.sp
                        )
                    }

                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    Text(
                        "${elapsedSeconds}s",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                    )
                }
            }

            is TranscribeState.Done -> {
                Column(modifier = Modifier.fillMaxSize()) {
                    TopAppBar(
                        title = { Text("Transcript") },
                        actions = {
                            TextButton(onClick = {
                                clipboard.setText(AnnotatedString(s.result.text))
                                Toast.makeText(context, "Copied", Toast.LENGTH_SHORT).show()
                            }) {
                                Text("Copy")
                            }
                            IconButton(onClick = {
                                val sendIntent = Intent(Intent.ACTION_SEND).apply {
                                    putExtra(Intent.EXTRA_TEXT, s.result.text)
                                    type = "text/plain"
                                }
                                context.startActivity(
                                    Intent.createChooser(sendIntent, "Share transcript")
                                )
                            }) {
                                Text("\u2197", fontSize = 20.sp)
                            }
                        }
                    )

                    SelectionContainer(
                        modifier = Modifier
                            .weight(1f)
                            .padding(16.dp)
                            .verticalScroll(rememberScrollState())
                    ) {
                        Text(
                            text = s.result.text,
                            style = MaterialTheme.typography.bodyLarge,
                            lineHeight = 26.sp
                        )
                    }

                    Text(
                        text = "%s · Transcribed in %.1fs".format(
                            languageDisplayName(s.result.detectedLanguage),
                            s.result.durationMs / 1000.0
                        ),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(horizontal = 16.dp)
                    )

                    Button(
                        onClick = onFinish,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp)
                    ) {
                        Text("Done")
                    }
                }
            }

            is TranscribeState.Error -> {
                Column(
                    modifier = Modifier.fillMaxSize(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Text("\u26A0", fontSize = 48.sp)
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        s.message,
                        style = MaterialTheme.typography.bodyLarge,
                        modifier = Modifier.padding(horizontal = 32.dp)
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Button(onClick = { retryTrigger++ }) {
                        Text("Try again")
                    }
                }
            }
        }
    }
}

private fun languageDisplayName(code: String): String {
    val names = mapOf(
        "en" to "English", "de" to "German", "fr" to "French",
        "es" to "Spanish", "it" to "Italian", "pt" to "Portuguese",
        "nl" to "Dutch", "pl" to "Polish", "ru" to "Russian",
        "zh" to "Chinese", "ja" to "Japanese", "ko" to "Korean",
        "ar" to "Arabic", "hi" to "Hindi", "tr" to "Turkish",
        "uk" to "Ukrainian", "cs" to "Czech", "sv" to "Swedish",
        "da" to "Danish", "fi" to "Finnish", "el" to "Greek",
        "he" to "Hebrew", "hu" to "Hungarian", "id" to "Indonesian",
        "ms" to "Malay", "no" to "Norwegian", "ro" to "Romanian",
        "sk" to "Slovak", "th" to "Thai", "vi" to "Vietnamese"
    )
    return names[code] ?: code.uppercase()
}
