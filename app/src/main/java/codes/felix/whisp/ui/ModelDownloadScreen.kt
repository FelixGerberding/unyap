package codes.felix.whisp.ui

import android.app.Activity
import android.view.WindowManager
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import codes.felix.whisp.whisper.WhisperEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

@Composable
fun ModelDownloadScreen(onReady: () -> Unit) {
    var progress by remember { mutableFloatStateOf(0f) }
    var downloading by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    var downloadedMb by remember { mutableIntStateOf(0) }
    var totalMb by remember { mutableIntStateOf(0) }
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // Keep the screen on while downloading: sleeping can drop the connection
    // mid-transfer on a ~190 MB download.
    DisposableEffect(downloading) {
        val window = (context as? Activity)?.window
        if (downloading) {
            window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
        onDispose { window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
    }

    fun startDownload() {
        downloading = true
        error = null
        progress = 0f
        downloadedMb = 0
        totalMb = 0

        scope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val destFile = File(context.filesDir, WhisperEngine.MODEL_FILENAME)
                    val tempFile = File(context.filesDir, "${WhisperEngine.MODEL_FILENAME}.tmp")

                    // Resume from a previous partial download if one exists.
                    val existing = if (tempFile.exists()) tempFile.length() else 0L

                    val conn = (URL(WhisperEngine.MODEL_URL).openConnection() as HttpURLConnection).apply {
                        instanceFollowRedirects = true
                        connectTimeout = 30_000
                        readTimeout = 30_000
                        if (existing > 0) setRequestProperty("Range", "bytes=$existing-")
                    }
                    conn.connect()

                    val code = conn.responseCode
                    if (code != HttpURLConnection.HTTP_OK && code != HttpURLConnection.HTTP_PARTIAL) {
                        throw IOException("Download failed: HTTP $code")
                    }
                    // 206 = server honored our range and we can append; 200 = full
                    // body, so any partial file must be overwritten from scratch.
                    val resuming = code == HttpURLConnection.HTTP_PARTIAL

                    val bodyBytes = conn.contentLengthLong  // remaining (206) or full (200), or -1
                    val totalBytes = if (resuming && bodyBytes > 0) existing + bodyBytes else bodyBytes
                    if (totalBytes > 0) totalMb = (totalBytes / 1_000_000).toInt()

                    // Make sure there's room before writing a few hundred MB.
                    val toWrite = if (resuming) bodyBytes else totalBytes
                    if (toWrite > 0 && context.filesDir.usableSpace < toWrite + 16L * 1024 * 1024) {
                        throw IOException("Not enough free space - need about ${toWrite / 1_000_000} MB free.")
                    }

                    var downloaded = if (resuming) existing else 0L
                    if (resuming) {
                        downloadedMb = (downloaded / 1_000_000).toInt()
                        if (totalBytes > 0) progress = downloaded.toFloat() / totalBytes
                    }

                    FileOutputStream(tempFile, resuming).use { output ->
                        conn.inputStream.buffered().use { input ->
                            val buffer = ByteArray(64 * 1024)
                            while (true) {
                                val bytesRead = input.read(buffer)
                                if (bytesRead == -1) break
                                output.write(buffer, 0, bytesRead)
                                downloaded += bytesRead
                                downloadedMb = (downloaded / 1_000_000).toInt()
                                if (totalBytes > 0) {
                                    progress = downloaded.toFloat() / totalBytes.toFloat()
                                }
                            }
                            output.fd.sync()
                        }
                    }

                    // Don't promote a truncated download to the real model file.
                    if (totalBytes > 0 && tempFile.length() != totalBytes) {
                        throw IOException("Download incomplete (${tempFile.length()} / $totalBytes bytes).")
                    }

                    destFile.delete()
                    if (!tempFile.renameTo(destFile)) {
                        // Fallback if the atomic rename ever fails.
                        tempFile.copyTo(destFile, overwrite = true)
                        tempFile.delete()
                    }
                }

                downloading = false
                onReady()
            } catch (e: Exception) {
                // Leave the .tmp in place so Retry resumes instead of restarting.
                downloading = false
                error = e.message ?: "Download failed"
            }
        }
    }

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
                "Setup Required",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                "Whisper Small model (~190 MB) - stored locally, used for all future transcriptions. " +
                    "No audio ever leaves your device.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.height(32.dp))

            if (downloading) {
                LinearProgressIndicator(
                    progress = progress,
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    if (totalMb > 0) "$downloadedMb MB / $totalMb MB (${(progress * 100).toInt()}%)"
                    else "$downloadedMb MB",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            } else if (error != null) {
                Text(
                    error!!,
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium
                )
                Spacer(modifier = Modifier.height(16.dp))
                Button(onClick = { startDownload() }) {
                    Text("Retry")
                }
            } else {
                Button(onClick = { startDownload() }) {
                    Text("Download Model")
                }
            }
        }
    }
}
