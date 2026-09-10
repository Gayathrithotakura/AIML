while ($true) {

    git add .

    $changes = git status --porcelain

    if ($changes) {
        git commit -m "Auto backup - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
        git push
    }

    Start-Sleep -Seconds 600
}