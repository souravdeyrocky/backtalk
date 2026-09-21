@echo off
REM Jarvis phone bridge -- Windows Firewall inbound rule, Tailscale-only.
REM
REM There is no LAN/home-Wi-Fi mode any more (Captain's call,
REM 2026-09-28) -- the phone bridge binds ONLY to this machine's
REM Tailscale interface IP, so this rule is scoped the same way:
REM
REM   - protocol=TCP localport=8765-8766  the cert-bootstrap port (8765)
REM                                       and the HTTPS app port (8766)
REM                                       -- change the numbers below to
REM                                       match phone.port/phone.tls_port
REM                                       in backtalk.json if you changed
REM                                       them from the defaults.
REM   - remoteip=100.64.0.0/10            ONLY Tailscale peers -- this is
REM                                       Tailscale's own shared CGNAT
REM                                       address range, not the internet
REM                                       and not your home LAN. A device
REM                                       has to be signed into your
REM                                       tailnet to have an address in
REM                                       this range at all.
REM   - localip=<your Tailscale IP>       run `tailscale ip -4` first and
REM                                       paste the result below (TS_IP)
REM                                       -- restricts this rule to
REM                                       traffic addressed to the
REM                                       Tailscale adapter specifically,
REM                                       not any other interface on this
REM                                       machine.
REM
REM NOT scoped by Windows network profile (private/public/domain) on
REM purpose: VPN virtual adapters are inconsistently classified across
REM Windows versions, and getting that wrong could silently block
REM legitimate Tailscale traffic. The remoteip + localip combination
REM above is the real, precise scope here.
REM
REM No port forwarding, no router changes, no internet exposure -- this
REM rule only ever affects traffic already arriving on the Tailscale
REM virtual adapter from another device on your own tailnet.
REM
REM Needs an ELEVATED (Administrator) command prompt AND Tailscale
REM already installed, running, and signed in (so `tailscale ip -4`
REM below returns a real address). Right-click this file and choose
REM "Run as administrator", or run it from an already-elevated
REM PowerShell/cmd window.

set PORTS=8765-8766

for /f "delims=" %%i in ('tailscale ip -4 2^>nul') do set TS_IP=%%i
if "%TS_IP%"=="" (
    echo Could not get a Tailscale IP ^(ran: tailscale ip -4^).
    echo Make sure Tailscale is installed, running, and signed in, then
    echo run this script again.
    pause
    exit /b 1
)
echo Detected Tailscale IP: %TS_IP%

netsh advfirewall firewall show rule name="Jarvis Phone Bridge" >nul 2>&1
if %ERRORLEVEL%==0 (
    echo A rule named "Jarvis Phone Bridge" already exists. Removing it
    echo first so this script is safe to re-run after a Tailscale IP or
    echo port change.
    netsh advfirewall firewall delete rule name="Jarvis Phone Bridge" >nul
)

netsh advfirewall firewall add rule ^
    name="Jarvis Phone Bridge" ^
    dir=in ^
    action=allow ^
    protocol=TCP ^
    localport=%PORTS% ^
    localip=%TS_IP% ^
    remoteip=100.64.0.0/10

if %ERRORLEVEL%==0 (
    echo.
    echo Done. Inbound TCP %PORTS% on %TS_IP% is now allowed from
    echo Tailscale peers only. Run phone-firewall-remove.bat to undo this.
) else (
    echo.
    echo FAILED -- this almost always means the prompt isn't elevated.
    echo Right-click phone-firewall-setup.bat and choose "Run as administrator".
)
pause
