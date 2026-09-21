@echo off
REM Removes the "Jarvis Phone Bridge" inbound firewall rule created by
REM phone-firewall-setup.bat. Needs an ELEVATED (Administrator) prompt.

netsh advfirewall firewall delete rule name="Jarvis Phone Bridge"

if %ERRORLEVEL%==0 (
    echo.
    echo Removed. Inbound access to the phone bridge port is no longer
    echo explicitly allowed by this rule.
) else (
    echo.
    echo Nothing to remove, or this prompt isn't elevated. Right-click
    echo this file and choose "Run as administrator" if you expected a
    echo rule to be there.
)
pause
