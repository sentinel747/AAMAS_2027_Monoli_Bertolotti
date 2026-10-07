"""Terminal UI primitives for the interactive headless configuration wizard.

Arrow-key navigation uses msvcrt on Windows and termios on POSIX. When stdin
or stdout is not a TTY (pipes, tests, CI) every menu degrades to a numbered
prompt read line-by-line, so the wizard stays fully scriptable.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

try:
    from rich.console import Console

    _console: Console | None = Console(highlight=False)
except ImportError:  # pragma: no cover - rich is an expected dependency
    _console = None


def interactive_capable() -> bool:
    """Arrow-key mode needs a real terminal on both ends; MARS_WIZARD_PLAIN forces the numbered fallback."""
    if os.environ.get("MARS_WIZARD_PLAIN"):
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


@dataclass
class MenuItem:
    label: str
    value: str
    hint: str = ""
    kind: str = "action"  # "action" | "back" | "separator"


@dataclass
class Menu:
    title: str
    items: list[MenuItem] = field(default_factory=list)
    subtitle: str = ""
    footer: str = "Frecce = muovi | Invio = conferma | numero = scelta rapida | Esc/0 = indietro"


def select_menu(menu: Menu, start_value: str | None = None) -> str | None:
    """Show a menu and return the chosen item's value, or None for back/exit."""
    selectable = [item for item in menu.items if item.kind != "separator"]
    if not selectable:
        return None
    if not interactive_capable():
        return _select_menu_plain(menu, selectable)
    index = 0
    if start_value is not None:
        for position, item in enumerate(selectable):
            if item.value == start_value:
                index = position
                break
    while True:
        _render_menu(menu, selectable, index)
        key = _read_key()
        if key == "up":
            index = (index - 1) % len(selectable)
        elif key == "down":
            index = (index + 1) % len(selectable)
        elif key == "home":
            index = 0
        elif key == "end":
            index = len(selectable) - 1
        elif key == "enter":
            chosen = selectable[index]
            return None if chosen.kind == "back" else chosen.value
        elif key in ("esc", "q"):
            return None
        elif key == "0":
            return None
        elif key.isdigit():
            position = int(key) - 1
            if 0 <= position < len(selectable):
                chosen = selectable[position]
                return None if chosen.kind == "back" else chosen.value


def prompt_text(label: str, *, default: str = "", note: str = "") -> str:
    """Line prompt; empty input keeps the default. EOF (piped stdin) keeps the default too."""
    if note:
        _print(f"  [dim]{note}[/dim]" if _console else f"  {note}")
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"{label}{suffix}: ").strip()
    except EOFError:
        return default
    return raw if raw else default


def prompt_number(
    label: str,
    *,
    default: float | int | None,
    minimum: float,
    maximum: float,
    integer: bool = True,
    optional: bool = False,
    note: str = "",
) -> float | int | None:
    """Numeric prompt clamped to [minimum, maximum]; '-' clears the value when optional."""
    if note:
        _print(f"  [dim]{note}[/dim]" if _console else f"  {note}")
    shown_default = "nessun limite" if optional and default is None else default
    while True:
        try:
            raw = input(f"{label} [{shown_default}]: ").strip()
        except EOFError:
            return default
        if not raw:
            return default
        if optional and raw in {"-", "none", "nessuno", "x"}:
            return None
        try:
            number = int(float(raw)) if integer else float(raw)
        except ValueError:
            _print("  Valore non valido, riprova (numero atteso).")
            continue
        clamped = min(maximum, max(minimum, number))
        if clamped != number:
            _print(f"  Valore portato nel range consentito: {clamped}")
        return int(clamped) if integer else float(clamped)


def confirm(label: str, *, default: bool = True) -> bool:
    suffix = "[S/n]" if default else "[s/N]"
    try:
        raw = input(f"{label} {suffix}: ").strip().lower()
    except EOFError:
        return default
    if not raw:
        return default
    return raw in {"s", "si", "sì", "y", "yes"}


def clear_screen() -> None:
    if not interactive_capable():
        return
    _enable_windows_vt_mode()
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


_vt_mode_enabled = False


def _enable_windows_vt_mode() -> None:
    """Turn on ANSI escape processing on legacy Windows consoles (no-op elsewhere)."""
    global _vt_mode_enabled
    if _vt_mode_enabled or os.name != "nt":
        return
    _vt_mode_enabled = True
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
    mode = ctypes.c_uint32()
    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING


def print_line(message: str = "") -> None:
    _print(message)


def print_heading(title: str) -> None:
    if _console:
        _console.print(f"[bold cyan]{title}[/bold cyan]")
    else:
        print(title)
        print("-" * len(title))


# --- internals ---------------------------------------------------------------


def _print(message: str) -> None:
    if _console:
        _console.print(message)
    else:
        print(message)


def _render_menu(menu: Menu, selectable: list[MenuItem], index: int) -> None:
    clear_screen()
    width = max((len(item.label) for item in menu.items), default=0)
    lines: list[tuple] = []
    position = 0
    for item in menu.items:
        if item.kind == "separator":
            lines.append(("separator", "", item.label, ""))
            continue
        position += 1
        selected = selectable[index] is item
        lines.append(("item", f"{position}", item, selected))
    if _console:
        _console.print(f"[bold cyan]{menu.title}[/bold cyan]")
        if menu.subtitle:
            _console.print(f"[dim]{menu.subtitle}[/dim]")
        _console.print()
        for kind, number, payload, selected in lines:
            if kind == "separator":
                label = str(payload)
                _console.print(f"  [dim]{label}[/dim]" if label else "")
                continue
            item = payload
            marker = "[bold yellow]>[/bold yellow]" if selected else " "
            label = item.label.ljust(width + 2)
            text = f" {marker} {number}. {label}"
            if item.hint:
                text += f"[dim]{item.hint}[/dim]"
            if selected:
                text = f"[bold]{text}[/bold]"
            _console.print(text)
        _console.print()
        _console.print(f"[dim]{menu.footer}[/dim]")
    else:  # pragma: no cover - rich is an expected dependency
        print(menu.title)
        if menu.subtitle:
            print(menu.subtitle)
        print()
        for kind, number, payload, selected in lines:
            if kind == "separator":
                print(f"  {payload}" if payload else "")
                continue
            item = payload
            marker = ">" if selected else " "
            print(f" {marker} {number}. {item.label.ljust(width + 2)}{item.hint}")
        print()
        print(menu.footer)


def _select_menu_plain(menu: Menu, selectable: list[MenuItem]) -> str | None:
    width = max((len(item.label) for item in selectable), default=0)
    while True:
        print()
        print(f"== {menu.title} ==")
        if menu.subtitle:
            print(menu.subtitle)
        position = 0
        for item in menu.items:
            if item.kind == "separator":
                if item.label:
                    print(f"  -- {item.label} --")
                continue
            position += 1
            hint = f"  {item.hint}" if item.hint else ""
            print(f"  {position}. {item.label.ljust(width)}{hint}")
        print("  0. Indietro")
        try:
            raw = input("Scelta: ").strip()
        except EOFError as exc:
            raise SystemExit("stdin chiuso: wizard interrotto senza avviare la run") from exc
        if raw in {"0", "q", "b"}:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(selectable):
            chosen = selectable[int(raw) - 1]
            return None if chosen.kind == "back" else chosen.value
        print("Scelta non valida.")


def _read_key() -> str:
    if os.name == "nt":
        return _read_key_windows()
    return _read_key_posix()


def _read_key_windows() -> str:
    import msvcrt

    char = msvcrt.getwch()
    if char in ("\x00", "\xe0"):
        code = msvcrt.getwch()
        return {"H": "up", "P": "down", "K": "left", "M": "right", "G": "home", "O": "end"}.get(code, "")
    if char in ("\r", "\n"):
        return "enter"
    if char == "\x1b":
        return "esc"
    if char == "\x03":
        raise KeyboardInterrupt
    return char


def _read_key_posix() -> str:  # pragma: no cover - exercised only on POSIX terminals
    import select
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        char = sys.stdin.read(1)
        if char == "\x1b":
            if select.select([sys.stdin], [], [], 0.05)[0]:
                sequence = sys.stdin.read(2)
                return {"[A": "up", "[B": "down", "[D": "left", "[C": "right", "[H": "home", "[F": "end"}.get(sequence, "esc")
            return "esc"
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)
    if char in ("\r", "\n"):
        return "enter"
    if char == "\x03":
        raise KeyboardInterrupt
    return char
