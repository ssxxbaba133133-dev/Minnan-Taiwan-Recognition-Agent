$baseUrl = $env:MODEL_API_BASE_URL
if (-not $baseUrl) { $baseUrl = $env:OPENAI_BASE_URL }
if (-not $baseUrl) { $baseUrl = $env:LMSTUDIO_BASE_URL }
if (-not $baseUrl) { $baseUrl = "http://127.0.0.1:1234/v1" }

$headers = @{}
$apiKey = $env:MODEL_API_KEY
if (-not $apiKey) { $apiKey = $env:OPENAI_API_KEY }
if (-not $apiKey) { $apiKey = $env:LMSTUDIO_API_KEY }
if ($apiKey) { $headers["Authorization"] = "Bearer $apiKey" }

Invoke-RestMethod -Uri "$($baseUrl.TrimEnd('/'))/models" -Headers $headers | ConvertTo-Json -Depth 8
