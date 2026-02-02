Write-Host "Initializing Git LFS."
git lfs install

Write-Host "Increasing Git HTTP buffer and disabling compression."
git config --global http.postBuffer 524288000
git config --global core.compression 0

Write-Host "Current LFS tracked files:"
git lfs track
git lfs ls-files

Write-Host "Forcing upload of all Git LFS objects."
git lfs push --all origin main

Write-Host "Retrying normal git push"
git push origin main

Write-Host " Done."
