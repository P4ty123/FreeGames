@echo off
setlocal
cd /d "%~dp0"
title CercaCrack
where py >nul 2>nul
if errorlevel 1 (echo Python non trovato: installalo da python.org & pause & exit /b 1)
set "PY=%TEMP%\cercacrack.py"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$l=Get-Content -LiteralPath '%~f0' -Encoding UTF8; [IO.File]::WriteAllLines('%PY%',$l[12..($l.Count-1)])"
py -3 "%PY%"
if errorlevel 1 pause
del "%PY%" >nul 2>nul
exit /b

# -*- coding: utf-8 -*-
import json, re, threading, time, urllib.parse, urllib.request, webbrowser, html as htmllib
import tkinter as tk
from tkinter import ttk, messagebox

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"

FONTI = {
    "FitGirl (repack)": "fitgirl-repacks.site",
    "DODI (repack)": "dodi-repacks.site",
    "CS.RIN.RU (crack)": "cs.rin.ru",
    "RuTracker": "rutracker.org",
    "Steam (titolo)": "store.steampowered.com",
}

def scarica(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")

def pulisci(t):
    t = re.sub(r"<[^>]+>", "", t)
    return htmllib.unescape(t).strip()

def cerca_ddg(query, site):
    q = urllib.parse.quote("site:" + site + " " + query)
    pagina = scarica("https://html.duckduckgo.com/html/?q=" + q)
    out = []
    for link, titolo in re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', pagina, re.S):
        if "uddg=" in link:
            link = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlparse(link).query).get("uddg", [""])[0])
        titolo = pulisci(titolo)
        if link and titolo:
            out.append((titolo, link))
    return out[:6]

def cerca_steam(query):
    url = "https://store.steampowered.com/api/storesearch/?cc=IT&l=italian&term=" + urllib.parse.quote(query)
    try:
        dati = json.loads(scarica(url))
    except Exception:
        return []
    return [(v.get("name", "?"), "https://store.steampowered.com/app/%s/" % v.get("id")) for v in dati.get("items", [])[:5]]

def vt_controlla(link, chiave):
    corpo = urllib.parse.urlencode({"url": link}).encode()
    req = urllib.request.Request("https://www.virustotal.com/api/v3/urls", data=corpo,
                                 headers={"x-apikey": chiave, "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        analisi = json.loads(r.read())["data"]["id"]
    for _ in range(6):
        time.sleep(5)
        req = urllib.request.Request("https://www.virustotal.com/api/v3/analyses/" + analisi,
                                     headers={"x-apikey": chiave, "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            stato = json.loads(r.read())["data"]["attributes"]
        if stato.get("status") == "completed":
            s = stato["stats"]
            return "%d malevoli su %d motori" % (s.get("malicious", 0), sum(s.values()))
    return "controllo non concluso"

class App:
    def __init__(self, root):
        self.root = root
        self.risultati = {}
        root.title("CercaCrack")
        root.geometry("820x560")
        alto = ttk.Frame(root, padding=10)
        alto.pack(fill="x")
        ttk.Label(alto, text="Gioco:").pack(side="left")
        self.q = ttk.Entry(alto)
        self.q.pack(side="left", fill="x", expand=True, padx=6)
        self.q.bind("<Return>", lambda e: self.cerca())
        ttk.Button(alto, text="Cerca", command=self.cerca).pack(side="left")
        opz = ttk.Frame(root, padding=(10, 0))
        opz.pack(fill="x")
        self.vars = {}
        for nome in FONTI:
            v = tk.BooleanVar(value=True)
            self.vars[nome] = v
            ttk.Checkbutton(opz, text=nome, variable=v).pack(side="left", padx=(0, 8))
        self.ita = tk.BooleanVar(value=False)
        ttk.Checkbutton(opz, text="Aggiungi 'ita'", variable=self.ita).pack(side="left")
        vt = ttk.Frame(root, padding=10)
        vt.pack(fill="x")
        ttk.Label(vt, text="Chiave VirusTotal (opzionale):").pack(side="left")
        self.chiave = ttk.Entry(vt, show="*")
        self.chiave.pack(side="left", fill="x", expand=True, padx=6)
        self.lista = ttk.Treeview(root, columns=("fonte",), height=14)
        self.lista.heading("#0", text="Risultato")
        self.lista.heading("fonte", text="Fonte")
        self.lista.column("#0", width=560)
        self.lista.column("fonte", width=180)
        self.lista.pack(fill="both", expand=True, padx=10)
        self.lista.bind("<Double-1>", self.apri)
        basso = ttk.Frame(root, padding=10)
        basso.pack(fill="x")
        ttk.Button(basso, text="Apri", command=self.apri).pack(side="left")
        ttk.Button(basso, text="Controlla su VirusTotal", command=self.controlla).pack(side="left", padx=6)
        self.stato = ttk.Label(basso, text="pronto")
        self.stato.pack(side="left", padx=10)

    def cerca(self):
        query = self.q.get().strip()
        if not query:
            return
        if self.ita.get():
            query += " ita"
        self.lista.delete(*self.lista.get_children())
        self.risultati.clear()
        attive = [n for n, v in self.vars.items() if v.get()]
        threading.Thread(target=self._cerca, args=(query, attive), daemon=True).start()

    def _cerca(self, query, attive):
        for nome in attive:
            self.stato.config(text="cerco su " + nome + "...")
            try:
                trovati = cerca_steam(query) if nome.startswith("Steam") else cerca_ddg(query, FONTI[nome])
            except Exception:
                self.stato.config(text=nome + ": errore")
                trovati = []
            for titolo, link in trovati:
                iid = self.lista.insert("", "end", text=titolo, values=(nome,))
                self.risultati[iid] = link
            time.sleep(1.2)
        self.stato.config(text="fatto: %d risultati" % len(self.risultati))

    def _sel(self):
        s = self.lista.selection()
        if not s:
            messagebox.showinfo("CercaCrack", "Seleziona prima un risultato.")
            return None
        return self.risultati.get(s[0])

    def apri(self, _=None):
        link = self._sel()
        if link:
            webbrowser.open(link)

    def controlla(self):
        link = self._sel()
        if not link:
            return
        chiave = self.chiave.get().strip()
        if not chiave:
            webbrowser.open("https://www.virustotal.com/gui/search/" + urllib.parse.quote(link, safe=""))
            return
        self.stato.config(text="controllo su VirusTotal...")
        threading.Thread(target=self._vt, args=(link, chiave), daemon=True).start()

    def _vt(self, link, chiave):
        try:
            esito = vt_controlla(link, chiave)
        except Exception as e:
            esito = "errore: %s" % e
        self.stato.config(text="VirusTotal: " + esito)

if __name__ == "__main__":
    r = tk.Tk()
    App(r)
    r.mainloop()
