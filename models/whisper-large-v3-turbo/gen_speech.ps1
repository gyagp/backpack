Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile("$PSScriptRoot\test_speech.wav")
$synth.Speak("Hello, this is a test of the Whisper speech recognition system running on WebGPU.")
$synth.Dispose()
Write-Host "Generated test_speech.wav"
