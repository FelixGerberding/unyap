# JNI bridge keep rules.
#
# The native library resolves its entry points by fully-qualified C symbol name
# (Java_codes_felix_whisp_whisper_WhisperEngine_initContext, ...) and calls back
# into Kotlin types by name via GetObjectClass/GetMethodID. R8 must not rename or
# strip any of these, or transcription crashes the moment minification is enabled.

# Keep every native method (and its declaring class name) intact.
-keepclasseswithmembernames,includedescriptorclasses class * {
    native <methods>;
}

# WhisperEngine declares the JNI entry points - keep the class and its members.
-keep class codes.felix.whisp.whisper.WhisperEngine {
    *;
}

# SegmentListener.onNewSegment(String) is invoked from native by name.
-keep class codes.felix.whisp.whisper.WhisperEngine$SegmentListener {
    *;
}
