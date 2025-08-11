# 📂 Historical Data [HistData](https://www.histdata.com)  

**1 minute EUR/USD data (Jan 2001 – July 2025)**  

## 📌 Source and Format  
The historical quotes were obtained from **[HistData](https://www.histdata.com)** in **Generic ASCII 1 minute** format:  
[http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/EURUSD](http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/EURUSD)  

---

## 📜 Transparency  

To ensure transparency and prove the authenticity of the original data, a multi-level verification and proof process was carried out:  

1. **Annual files** (CSV with minute data and TXT reports) have their own **SHA-256 hashes** and **GPG signatures** (`.sig`/`asc`).  
2. A **single archive** `source_data_2001-2025.tar.xz` contains all annual data and also has its own **SHA-256 hash**, **GPG signature**, and an **OTS anchor** (OpenTimestamps) in the blockchain.  
3. A **Git repository with Verified commits (GPG)** records the history of changes and protects against file tampering.  

🔍 **Purpose:**  
- **SHA-256 hash** — verifies file integrity.  
- **GPG signature** — confirms authorship and authenticity.  
- **OTS anchor** — blockchain proof of the data’s existence at a specific point in time.  

⚠️ **Important:**  
- The data was taken from **[HistData](https://www.histdata.com)** at the time it was available in its original form.  
- **HistData** may update its history (including removing gaps and modifying quotes).  
- The hashes, signatures, and OTS prove that the backtest was conducted **on this exact dataset**, not on any altered later version.  
