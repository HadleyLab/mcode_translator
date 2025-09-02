# UI with NiceGUI
- UI must be implemented **exclusively in Python** using [NiceGUI](https://nicegui.io/).
- No custom JavaScript, CSS, or event wiring; rely only on NiceGUI's built-in components.
- Keep UI logic thin; domain logic remains in service layer (pure Python).
- Fail fast: missing UI state or config must raise explicit exceptions.
- Use strict typing and dataclasses/Pydantic for all UI-bound data.
- Layouts: prefer container/grid APIs from NiceGUI; avoid custom positioning hacks.
- Notifications, toasts, and dialogs: only use NiceGUI’s standard components.
- Infrastructure-first: no convenience shortcuts; always explicitly declare elements and bindings.
