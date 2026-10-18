package codes.felix.whisp.audio

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

object OpusDecoder {

    init {
        System.loadLibrary("whisperjni")
    }

    private const val TARGET_SAMPLE_RATE = 16000
    private const val TIMEOUT_US = 10_000L

    suspend fun decodeToFloatArray(context: Context, uri: Uri): FloatArray =
        withContext(Dispatchers.Default) {
            val extractor = MediaExtractor()
            try {
                extractor.setDataSource(context, uri, null)
            } catch (e: Exception) {
                throw IOException("Cannot read audio from URI: $uri", e)
            }

            val trackIndex = findAudioTrack(extractor)
                ?: throw IOException("No audio track found in file")
            extractor.selectTrack(trackIndex)

            val format = extractor.getTrackFormat(trackIndex)
            val mime = format.getString(MediaFormat.KEY_MIME)
                ?: throw IOException("No MIME type in audio track")

            val opusHead = format.getByteBuffer("csd-0")?.order(ByteOrder.LITTLE_ENDIAN)
            if (mime == MediaFormat.MIMETYPE_AUDIO_OPUS && opusHead != null &&
                opusHead.remaining() >= 19 && opusHead.get(opusHead.position() + 9) <= 2
            ) {
                try {
                    return@withContext decodeOpusNatively(extractor, opusHead)
                } finally {
                    extractor.release()
                }
            }

            val codec = try {
                MediaCodec.createDecoderByType(mime)
            } catch (e: Exception) {
                throw IOException("Unsupported audio format: $mime", e)
            }

            codec.configure(format, null, null, 0)
            codec.start()

            // Growable primitive PCM buffer. A MutableList<Short> boxes every
            // sample (~16 bytes each), which OOM-crashes on long recordings; a
            // raw ShortArray keeps it at 2 bytes/sample.
            var pcm = ShortArray(1 shl 18)  // 256K samples (~0.5 MB) to start
            var pcmLen = 0
            fun ensureCapacity(extra: Int) {
                if (pcmLen + extra > pcm.size) {
                    var newSize = pcm.size
                    while (newSize < pcmLen + extra) newSize = newSize shl 1
                    pcm = pcm.copyOf(newSize)
                }
            }
            var outputSampleRate = TARGET_SAMPLE_RATE
            var outputChannels = 1
            val bufferInfo = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false

            try {
                while (!outputDone) {
                    // Feed input without waiting, so the codec's input queue
                    // stays full. Blocking here (or on output while input is
                    // still available) costs up to TIMEOUT_US per packet, which
                    // for 20 ms Opus packets adds seconds per minute of audio.
                    var fed = false
                    if (!inputDone) {
                        val inputIndex = codec.dequeueInputBuffer(0)
                        if (inputIndex >= 0) {
                            val inputBuffer = codec.getInputBuffer(inputIndex)!!
                            val sampleSize = extractor.readSampleData(inputBuffer, 0)
                            if (sampleSize < 0) {
                                codec.queueInputBuffer(
                                    inputIndex, 0, 0, 0,
                                    MediaCodec.BUFFER_FLAG_END_OF_STREAM
                                )
                                inputDone = true
                            } else {
                                val pts = extractor.sampleTime
                                codec.queueInputBuffer(inputIndex, 0, sampleSize, pts, 0)
                                extractor.advance()
                            }
                            fed = true
                        }
                    }

                    // Drain output; only wait for it when there was nothing to feed
                    val outputIndex = codec.dequeueOutputBuffer(bufferInfo, if (fed) 0 else TIMEOUT_US)
                    when {
                        outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            val outFormat = codec.outputFormat
                            outputSampleRate = outFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                            outputChannels = outFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        }
                        outputIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                            // No output available yet
                        }
                        outputIndex >= 0 -> {
                            val outputBuffer = codec.getOutputBuffer(outputIndex)!!
                            val shortBuffer = outputBuffer.order(ByteOrder.nativeOrder())
                                .asShortBuffer()
                            val samples = ShortArray(shortBuffer.remaining())
                            shortBuffer.get(samples)

                            // Mix to mono if stereo
                            if (outputChannels > 1) {
                                ensureCapacity(samples.size / outputChannels + 1)
                                for (i in samples.indices step outputChannels) {
                                    var sum = 0L
                                    for (ch in 0 until outputChannels) {
                                        if (i + ch < samples.size) {
                                            sum += samples[i + ch]
                                        }
                                    }
                                    pcm[pcmLen++] = (sum / outputChannels).toShort()
                                }
                            } else {
                                ensureCapacity(samples.size)
                                System.arraycopy(samples, 0, pcm, pcmLen, samples.size)
                                pcmLen += samples.size
                            }

                            codec.releaseOutputBuffer(outputIndex, false)

                            if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                                outputDone = true
                            }
                        }
                    }
                }
            } finally {
                codec.stop()
                codec.release()
                extractor.release()
            }

            // Convert to float and resample if needed
            val monoFloat = FloatArray(pcmLen) { pcm[it] / 32768f }

            if (outputSampleRate != TARGET_SAMPLE_RATE) {
                resample(monoFloat, outputSampleRate, TARGET_SAMPLE_RATE)
            } else {
                monoFloat
            }
        }

    /**
     * Demuxes all packets and decodes them with libopus, straight to 16 kHz mono.
     * Much faster than MediaCodec, which costs an IPC round trip per 20 ms packet.
     * [opusHead] is the OpusHead header (csd-0); mono or stereo streams only.
     */
    private fun decodeOpusNatively(extractor: MediaExtractor, opusHead: ByteBuffer): FloatArray {
        val preSkip = opusHead.getShort(opusHead.position() + 10).toInt() and 0xFFFF
        var data = ByteArray(1 shl 16)
        var dataLen = 0
        var sizes = IntArray(1 shl 12)
        var n = 0
        val buf = ByteBuffer.allocate(1 shl 16)  // far larger than any Opus packet
        while (true) {
            buf.clear()
            val size = extractor.readSampleData(buf, 0)
            if (size < 0) break
            if (dataLen + size > data.size) data = data.copyOf(maxOf(data.size * 2, dataLen + size))
            if (n == sizes.size) sizes = sizes.copyOf(n * 2)
            buf.position(0)
            buf.limit(size)
            buf.get(data, dataLen, size)
            dataLen += size
            sizes[n++] = size
            extractor.advance()
        }
        return decodeOpusNative(data, sizes.copyOf(n), preSkip)
            ?: throw IOException("Opus decoding failed")
    }

    private external fun decodeOpusNative(packets: ByteArray, sizes: IntArray, preSkip: Int): FloatArray?

    private fun findAudioTrack(extractor: MediaExtractor): Int? {
        for (i in 0 until extractor.trackCount) {
            val mime = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) return i
        }
        return null
    }

    private fun resample(input: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        val ratio = fromRate.toDouble() / toRate.toDouble()
        val outputLen = (input.size / ratio).toInt()
        val output = FloatArray(outputLen)
        for (i in output.indices) {
            val srcPos = i * ratio
            val srcIndex = srcPos.toInt()
            val frac = (srcPos - srcIndex).toFloat()
            output[i] = if (srcIndex + 1 < input.size) {
                input[srcIndex] * (1f - frac) + input[srcIndex + 1] * frac
            } else {
                input[srcIndex.coerceAtMost(input.size - 1)]
            }
        }
        return output
    }
}
