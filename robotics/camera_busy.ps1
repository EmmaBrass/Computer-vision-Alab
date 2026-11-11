# Exit 1 if another process is holding the lab Orbbec cameras.
$hits = Get-CimInstance Win32_Process | Where-Object {
    $c = $_.CommandLine
    $n = $_.Name
    if (-not $c) { return $false }
    if ($c -match 'Get-CimInstance|camera_busy\.ps1|record_walkthrough') { return $false }
    ($n -match '^(orbbec_camera_publisher|OrbbecViewer)(\.exe)?$') -or
    ($c -match 'orbbec_camera_publisher|orbbec_dual_sync|OrbbecViewer|view_dual_cameras\.py|scan_lab_cloud\.py')
}
if ($hits) {
    $hits | ForEach-Object {
        Write-Host ("HOLDING CAMERAS  PID {0}  {1}" -f $_.ProcessId, $_.Name)
        Write-Host $_.CommandLine
    }
    exit 1
}
exit 0
