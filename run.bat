@echo off
REM ============================================================
REM  LifeStream - chay video voi model da train
REM  Cach dung:  run.bat                -> menu
REM              run.bat flood data\ten_video.mp4   (chi dinh video)
REM              run.bat flood          -> tu chon video ngau nhien trong data\
REM              run.bat pool  <video>  /  run.bat pool (ngau nhien)
REM              run.bat 6              -> chon video cu the trong data\
REM              run.bat sim            -> mo phong ho boi
REM              run.bat fsim           -> mo phong lu
REM  Luu y: chon 1 hoac 2 trong menu se tu phat MOT VIDEO NGAU NHIEN
REM         trong thu muc data\ (khong can nhap duong dan). Chon 6 de
REM         tu tay chon video trong danh sach data\.
REM  LUU Y: file nay PHAI dung xuong dong CRLF, neu doi sang LF
REM         cmd.exe se bao "The syntax of the command is incorrect".
REM ============================================================
setlocal
cd /d "%~dp0"
title LifeStream

where python >nul 2>nul
if errorlevel 1 goto no_python

REM ---- chay truc tiep: run.bat flood <video> ----
if "%~1"=="" goto menu_start
if /i "%~1"=="flood" goto flood
if /i "%~1"=="pool" goto pool
if "%~1"=="6" goto pick_menu
if /i "%~1"=="sim" goto sim
if /i "%~1"=="fsim" goto fsim

REM ---- menu: toi da 5 lan nhap sai roi tu thoat ----
:menu_start
set "tries=0"

:menu
cls
echo.
echo  ============ LifeStream ============
echo   1. Phat hien lu cuon - video ngau nhien (flood)
echo   2. Phat hien duoi nuoc - video ngau nhien (pool)
echo   3. Mo phong ho boi - demo
echo   4. Mo phong lu - demo
echo   5. Quay video lu thanh file mp4 - demo
echo   6. Chon video cu the trong data\
echo   Q. Thoat
echo =====================================
echo   Chay truc tiep: run.bat flood data\ten_video.mp4
echo.
set "choice="
set /p "choice=Chon 1-6 hoac Q: "
if not defined choice goto quit
if /i "%choice%"=="q" goto quit
if "%choice%"=="1" goto menu_flood
if "%choice%"=="2" goto menu_pool
if "%choice%"=="3" goto sim
if "%choice%"=="4" goto fsim
if "%choice%"=="5" goto record_demo
if "%choice%"=="6" goto pick_menu
set /a tries+=1
if %tries% geq 5 goto quit
echo.
echo Lua chon '%choice%' khong hop le - nhap 1-6 hoac Q.
goto menu

REM ------------------------------------------------------------
REM  Menu 1/2 -> tu chon video ngau nhien trong data\
REM ------------------------------------------------------------
:menu_flood
call :pick_random
if errorlevel 1 goto menu
call :run_flood
goto end

:menu_pool
call :pick_random
if errorlevel 1 goto menu
call :run_pool
goto end

REM ------------------------------------------------------------
REM  Menu 6 -> liet ke video trong data\ de nguoi dung chon
REM ------------------------------------------------------------
:pick_menu
cls
echo.
echo  ---------- Video trong data\ -----------
call :pick_list
if errorlevel 1 goto menu
echo  ----------------------------------------
echo  (Nhap so thu tu, Q de quay lai menu)
echo.
set "PIDX="
set /p "PIDX=Chon video so: "
if not defined PIDX goto menu
if /i "%PIDX%"=="q" goto menu
set "LIFESTREAM_PICK_INDEX=%PIDX%"
call :pick_get
set "LIFESTREAM_PICK_INDEX="
if errorlevel 1 goto pick_menu
call :pick_mode
if errorlevel 1 goto menu
goto end

REM ------------------------------------------------------------
REM  Truc tiep: run.bat flood [video] - thieu video = chon ngau nhien
REM ------------------------------------------------------------
:flood
set "VID=%~2"
if defined VID goto flood_go
call :pick_random
if errorlevel 1 exit /b 1
:flood_go
call :run_flood
goto end

:pool
set "VID=%~2"
if defined VID goto pool_go
call :pick_random
if errorlevel 1 exit /b 1
:pool_go
call :run_pool
goto end

REM ------------------------------------------------------------
REM  Mo phong - khong can camera
REM ------------------------------------------------------------
:sim
python -m lifestream --scorer learned --simulate --scenario mixed --seconds 30
goto end

:fsim
python -m lifestream --mode flood --scorer learned --simulate --flood-scenario drift --seconds 30
goto end

:record_demo
python -m lifestream --mode flood --scorer learned --simulate --flood-scenario drift --seconds 30 --record demo\flood_menu_demo.mp4
goto end

REM ------------------------------------------------------------
REM  Ham dung chung
REM ------------------------------------------------------------
REM In danh sach video so thu tu (i|duong_dan). Loi -> exit /b 1.
:pick_list
python scripts\pick_video.py --list
if errorlevel 1 (
    echo [LOI] Khong doc duoc danh sach video trong thu muc data\.
    pause
    exit /b 1
)
exit /b 0

REM Doc LIFESTREAM_PICK_INDEX -> set VID. Loi -> exit /b 1 (thong bao tu python).
:pick_get
set "VID="
for /f "usebackq delims=" %%F in (`python scripts\pick_video.py --path`) do set "VID=%%F"
if not defined VID exit /b 1
exit /b 0

REM Chon ngau nhien: set VID thanh duong dan .mp4 trong data\.
REM  Loi (khong co video / thieu thu muc data) -> exit /b 1.
:pick_random
set "VID="
for /f "usebackq delims=" %%F in (`python scripts\pick_video.py`) do set "VID=%%F"
if not defined VID (
    echo [LOI] Khong lay duoc video ngau nhien tu thu muc data\.
    pause
    exit /b 1
)
exit /b 0

REM Hoi mode cho video da chon o menu 6: f = flood, p = pool, khac = bo qua.
:pick_mode
echo Da chon: "%VID%"
set "PMODE="
set /p "PMODE=Chay f=flood lu / p=pool boi / Enter de bo qua: "
if /i "%PMODE%"=="f" call :run_flood & exit /b 0
if /i "%PMODE%"=="p" call :run_pool & exit /b 0
exit /b 0

:run_flood
echo [LifeStream] Flood Guard dang chay: "%VID%"
python -m lifestream --mode flood --scorer learned --source "%VID%"
exit /b 0

:run_pool
echo [LifeStream] Pool Guard dang chay: "%VID%"
python -m lifestream --mode pool --scorer learned --source "%VID%"
exit /b 0

:no_python
echo [LOI] Khong tim thay python trong PATH.
echo       Cai Python 3.10+ va them thu muc Scripts vao PATH, roi thu lai.
pause
exit /b 1

:quit
echo.
echo Tam biet.
exit /b 0

:end
echo.
pause
exit /b 0
