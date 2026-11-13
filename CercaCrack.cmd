@echo off
setlocal
cd /d "%~dp0"
title CercaCrack
where py >nul 2>nul
if errorlevel 1 (echo Python non trovato: installalo da python.org & pause & exit /b 1)
set "CC_SRC=%~f0"
set "CC_PY=%TEMP%\cercacrack_%RANDOM%.py"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$l=@(Get-Content -LiteralPath $env:CC_SRC -Encoding UTF8); $i=[Array]::IndexOf($l,'#__PYTHON__'); if($i -lt 0){exit 1}; [IO.File]::WriteAllLines($env:CC_PY,[string[]]$l[($i+1)..($l.Count-1)])"
if errorlevel 1 (echo Impossibile estrarre lo script Python. & pause & exit /b 1)
py -3 "%CC_PY%"
if errorlevel 1 pause
del "%CC_PY%" >nul 2>nul
exit /b

#__PYTHON__
# -*- coding: utf-8 -*-
import json, queue, re, threading, time, urllib.error, urllib.parse, urllib.request, webbrowser, html as htmllib
import tkinter as tk
from tkinter import ttk, messagebox

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"

FONTI = {
    "FitGirl (repack)": "fitgirl-repacks.site",
    "DODI (repack)": "dodi-repacks.site",
    "CS.RIN.RU (crack)": "cs.rin.ru",
    "RuTracker": "rutracker.org",
    "The Pirate Bay": "thepiratebay.org",
    "Cracked-Games (rischioso)": "cracked-games.org",
    "Steam (titolo)": "store.steampowered.com",
}

# Fonti che compaiono nella finestra ma partono disattivate (spuntale a mano)
DISATTIVATE = {"Cracked-Games (rischioso)"}

def scarica(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")

def pulisci(t):
    t = re.sub(r"<[^>]+>", "", t)
    return htmllib.unescape(t).strip()

BLOCCO = re.compile(r"anomaly|captcha|challenge-form|bots use DuckDuckGo", re.I)

def estrai_link(pagina, classe):
    out = []
    for attrs, testo in re.findall(r"<a\s([^>]*)>(.*?)</a>", pagina, re.S):
        if not re.search(r"class=[\"'][^\"']*\b%s\b" % classe, attrs):
            continue
        m = re.search(r"href=[\"']([^\"']+)", attrs)
        if not m:
            continue
        link = htmllib.unescape(m.group(1))
        if "uddg=" in link:
            link = urllib.parse.parse_qs(urllib.parse.urlparse(link).query).get("uddg", [""])[0]
        if link.startswith("//"):
            link = "https:" + link
        titolo = pulisci(testo)
        if link.startswith("http") and "duckduckgo.com" not in link and titolo:
            out.append((titolo, link))
    return out

def cerca_ddg(query, site):
    """Restituisce (risultati, bloccato)."""
    q = urllib.parse.quote("site:" + site + " " + query)
    bloccato = False
    for url, classe in (("https://html.duckduckgo.com/html/?q=" + q, "result__a"),
                        ("https://lite.duckduckgo.com/lite/?q=" + q, "result-link")):
        try:
            pagina = scarica(url)
        except urllib.error.HTTPError:
            bloccato = True
            continue
        trovati = estrai_link(pagina, classe)
        if trovati:
            return trovati[:6], False
        if BLOCCO.search(pagina):
            bloccato = True
        else:
            return [], False
    return [], bloccato

def link_browser(query, site):
    return "https://duckduckgo.com/?q=" + urllib.parse.quote("site:" + site + " " + query)

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
        self.coda = queue.Queue()
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
        for i, nome in enumerate(FONTI):
            v = tk.BooleanVar(value=nome not in DISATTIVATE)
            self.vars[nome] = v
            ttk.Checkbutton(opz, text=nome, variable=v).grid(row=i // 4, column=i % 4, sticky="w", padx=(0, 12), pady=1)
        self.ita = tk.BooleanVar(value=False)
        n = len(FONTI)
        ttk.Checkbutton(opz, text="Aggiungi 'ita'", variable=self.ita).grid(row=n // 4, column=n % 4, sticky="w", pady=1)
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
        self._svuota_coda()

    def ui(self, funz, *args):
        """Chiamabile da qualsiasi thread: esegue funz nel thread di Tkinter."""
        self.coda.put((funz, args))

    def _svuota_coda(self):
        try:
            while True:
                funz, args = self.coda.get_nowait()
                funz(*args)
        except queue.Empty:
            pass
        self.root.after(100, self._svuota_coda)

    def _imposta_stato(self, testo):
        self.stato.config(text=testo)

    def _aggiungi(self, titolo, nome, link):
        iid = self.lista.insert("", "end", text=titolo, values=(nome,))
        self.risultati[iid] = link

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
        bloccate = 0
        totale = 0
        for nome in attive:
            self.ui(self._imposta_stato, "cerco su " + nome + "...")
            bloccato = False
            try:
                if nome.startswith("Steam"):
                    trovati = cerca_steam(query)
                else:
                    trovati, bloccato = cerca_ddg(query, FONTI[nome])
            except Exception:
                trovati = []
                bloccato = True
            for titolo, link in trovati:
                self.ui(self._aggiungi, titolo, nome, link)
            totale += len(trovati)
            if bloccato and not nome.startswith("Steam"):
                bloccate += 1
                self.ui(self._aggiungi, "[!] Ricerca bloccata: doppio clic per cercare nel browser",
                        nome, link_browser(query, FONTI[nome]))
            time.sleep(1.2)
        msg = "fatto: %d risultati" % totale
        if bloccate:
            msg += " (%d fonti bloccate, apri le righe [!])" % bloccate
        self.ui(self._imposta_stato, msg)

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
        self.ui(self._imposta_stato, "VirusTotal: " + esito)

if __name__ == "__main__":
    r = tk.Tk()
    App(r)
    r.mainloop()
