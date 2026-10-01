# Docker

Development and CI container definitions for GesturePilot.

**Scope:** linting, type checks, and headless test runs on Linux.

**Not a runtime target.** Camera capture, Windows APIs, and the system tray component
require a native Windows host and cannot run in a container. The application itself is
always run natively on Windows.

No container definition is included yet — nothing in this phase needs one. A
`Dockerfile` will be added when there is a concrete CI job to support.