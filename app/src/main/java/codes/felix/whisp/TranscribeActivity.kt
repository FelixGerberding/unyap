package codes.felix.whisp

import android.content.Intent
import android.content.res.Configuration
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import codes.felix.whisp.ui.ModelDownloadScreen
import codes.felix.whisp.ui.TranscriptScreen
import codes.felix.whisp.whisper.WhisperEngine

class TranscribeActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val launchUri: Uri? = when (intent?.action) {
            Intent.ACTION_SEND -> {
                @Suppress("DEPRECATION")
                intent.getParcelableExtra(Intent.EXTRA_STREAM)
            }
            else -> null
        }

        // Launched from the home screen (not a share): make the window opaque.
        // Match the system theme so there's no light flash in dark mode.
        if (launchUri == null) {
            val isDark = (resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK) ==
                Configuration.UI_MODE_NIGHT_YES
            window.setBackgroundDrawableResource(
                if (isDark) android.R.color.black else android.R.color.white
            )
            window.setDimAmount(0f)
        }

        val engine = try {
            WhisperEngine.getInstance(this)
        } catch (e: UnsatisfiedLinkError) {
            finish()
            return
        }
        // Load the model while the shared audio is being decoded.
        engine.preload()

        setContent {
            val darkTheme = isSystemInDarkTheme()
            MaterialTheme(colorScheme = if (darkTheme) darkColorScheme() else lightColorScheme()) {
                var modelReady by remember { mutableStateOf(engine.modelReady) }
                var pickedUri by remember { mutableStateOf<Uri?>(null) }

                val picker = rememberLauncherForActivityResult(
                    ActivityResultContracts.OpenDocument()
                ) { uri -> if (uri != null) pickedUri = uri }

                val activeUri = pickedUri ?: launchUri

                when {
                    !modelReady -> ModelDownloadScreen(onReady = {
                        engine.refreshModelState()
                        modelReady = true
                    })

                    activeUri != null -> TranscriptScreen(
                        audioUri = activeUri,
                        engine = engine,
                        // Picked in-app: go back home for another. Shared in: close.
                        onFinish = { if (pickedUri != null) pickedUri = null else finish() }
                    )

                    else -> HomeScreen(onPickFile = { picker.launch(arrayOf("audio/*")) })
                }
            }
        }
    }
}

@Composable
private fun HomeScreen(onPickFile: () -> Unit) {
    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                "Unyap",
                style = MaterialTheme.typography.headlineLarge
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                "Read any voice note",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(32.dp))
            Text(
                "Turn a long voice note into text you can read, right on your phone.",
                style = MaterialTheme.typography.bodyLarge,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(24.dp))
            Button(onClick = onPickFile) {
                Text("Select audio file")
            }
            Spacer(modifier = Modifier.height(24.dp))
            Text(
                "Tip: share a voice note straight from WhatsApp or any other app into Unyap.",
                style = MaterialTheme.typography.bodyMedium,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.outline
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                "All processing happens on-device. No audio or text ever leaves your phone.",
                style = MaterialTheme.typography.bodyMedium,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.outline
            )
        }
    }
}
