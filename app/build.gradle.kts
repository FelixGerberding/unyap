import java.io.FileInputStream
import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Signing credentials are kept out of version control. Copy
// keystore.properties.template to keystore.properties (gitignored) and fill it in.
val keystorePropertiesFile = rootProject.file("keystore.properties")
val keystoreProperties = Properties().apply {
    if (keystorePropertiesFile.exists()) {
        load(FileInputStream(keystorePropertiesFile))
    }
}

android {
    namespace = "codes.felix.whisp"
    compileSdk = 36

    defaultConfig {
        applicationId = "codes.felix.whisp"
        minSdk = 29
        targetSdk = 36
        versionCode = 4
        versionName = "1.1"

        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    signingConfigs {
        create("release") {
            if (keystorePropertiesFile.exists()) {
                storeFile = file(keystoreProperties.getProperty("storeFile"))
                storePassword = keystoreProperties.getProperty("storePassword")
                keyAlias = keystoreProperties.getProperty("keyAlias")
                keyPassword = keystoreProperties.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        debug {
            // Installs next to the Play Store build instead of clashing with it.
            applicationIdSuffix = ".dev"
            versionNameSuffix = "-dev"
        }
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            // Only attach the signing config when credentials are present, so
            // checkouts without the keystore can still configure/build.
            if (keystorePropertiesFile.exists()) {
                signingConfig = signingConfigs.getByName("release")
            }
            // Ship full native debug symbols in the bundle so Play Console can
            // symbolicate native (ggml/whisper .so) crashes.
            ndk {
                debugSymbolLevel = "FULL"
            }
        }
        // Release-equivalent build with a benchmark entry point (src/benchmark),
        // installable next to the other builds. See BenchActivity.
        create("benchmark") {
            initWith(getByName("release"))
            applicationIdSuffix = ".bench"
            versionNameSuffix = "-bench"
            signingConfig = signingConfigs.getByName("debug")
            matchingFallbacks += "release"

        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    buildFeatures {
        compose = true
    }

    externalNativeBuild {
        cmake {
            path = file("CMakeLists.txt")
            version = "3.22.1"
        }
    }

    packaging {
        jniLibs {
            // ggml loads its per-CPU backend variants (libggml-cpu-android_*.so) by
            // enumerating the native library directory at runtime. That only works if
            // the .so files are extracted to disk, so opt out of the modern
            // compressed-in-APK packaging.
            useLegacyPackaging = true
            // Don't ship ggml's SVE/SME CPU variants (armv9.x). ggml scores them
            // highest, but on 128-bit SVE (as in all current phone cores) the
            // armv9.0 variant ran the encoder ~3x slower than the NEON+i8mm
            // armv8.6 one. jni_bridge.cpp only loads the NEON variants anyway.
            excludes += "**/libggml-cpu-android_armv9*.so"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.activity:activity-compose:1.8.2")
    implementation(platform("androidx.compose:compose-bom:2024.09.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.foundation:foundation")
    implementation("com.google.android.material:material:1.11.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
}
