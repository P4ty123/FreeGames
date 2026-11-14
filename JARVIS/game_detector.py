import time
import psutil


def get_processes():
    processes = {}

    for process in psutil.process_iter(["pid", "name", "exe"]):
        try:
            pid = process.info["pid"]
            name = process.info["name"]
            exe = process.info["exe"]

            processes[pid] = {
                "name": name,
                "exe": exe,
            }

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return processes


print("=" * 60)
print("JARVIS - GAME DETECTOR")
print("=" * 60)
print()
print("1. Lascia aperto questo programma.")
print("2. Avvia UNO dei giochi che vuoi verificare.")
print("3. Aspetta che il gioco sia completamente avviato.")
print("4. Torna qui e premi INVIO.")
print()

before = get_processes()

input("Premi INVIO quando il gioco è aperto...")

time.sleep(2)

after = get_processes()

new_processes = []

for pid, info in after.items():
    if pid not in before:
        new_processes.append(
            (
                pid,
                info["name"],
                info["exe"],
            )
        )


print()
print("=" * 60)
print("NUOVI PROCESSI RILEVATI")
print("=" * 60)
print()

if not new_processes:
    print("Nessun nuovo processo rilevato.")
else:
    for pid, name, exe in new_processes:
        print(f"PID : {pid}")
        print(f"Nome: {name}")
        print(f"EXE : {exe}")
        print("-" * 60)

print()
input("Premi INVIO per chiudere...")