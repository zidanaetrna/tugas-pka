"""Halaman hasil (hasil/index.html) dan halaman status saat eksperimen berjalan."""
import html
import math
import os

STYLE = """
:root{--bg:#f7f7f5;--fg:#1d1d1f;--mut:#5f6368;--line:#d9d9d6;--head:#ececea;--acc:#2563a8;--bad:#c0461b}
@media (prefers-color-scheme:dark){:root{--bg:#18191b;--fg:#e8e8e6;--mut:#a0a3a8;--line:#3a3b3e;--head:#26272a;--acc:#7fb0e8;--bad:#f08a63}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 "Times New Roman",Georgia,serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 10px;text-transform:uppercase;letter-spacing:.02em}
p.mut{color:var(--mut);margin:0 0 18px}
.tbl{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:center;white-space:nowrap}
th{background:var(--head)}td:first-child,th:first-child{text-align:left}
caption{caption-side:top;text-align:center;padding:0 0 8px;font-size:14px}caption b{margin-right:4px}
img{max-width:100%;background:#fff;border:1px solid var(--line)}
.bad{color:var(--bad)}.note{font-size:13px;color:var(--mut);margin-top:6px}
pre{background:var(--head);padding:12px;overflow-x:auto;font-size:13px}
"""


def _num(x, d):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt(v, col=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "Median k" in col:
            return _num(v, 0)
        if "Selisih" in col or "Kenaikan" in col:
            return ("+" if v > 0 else "") + _num(v, 2).replace("-", "−")
        if "(pp)" in col:
            return _num(v, 2)
        if "(%)" in col:
            return _num(v, 2)
        return _num(v, 3)
    return html.escape(str(v))


def _table(rows, cap):
    cols = list(rows[0].keys())
    h = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    b = "".join("<tr>" + "".join(f"<td>{_fmt(r[c], c)}</td>" for c in cols) + "</tr>" for r in rows)
    return f'<div class="tbl"><table><caption>{cap}</caption><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def write_html(r, out):
    i = r["info"]
    nol = ", ".join(f"{k} {v}" for k, v in i["nol_per_variabel"].items())
    leak = r["median_per_kelas_tidak_valid"]
    body = f"""
<h1>Hasil Eksperimen: Penanganan Missing Values pada KNN + Grid Search</h1>
<p class="mut">Dataset diabetes, {i['baris']} baris ({i['positif']} positif, {i['persen_positif']:.1f}%).
Nested CV {i['outer']}, inner {i['inner']}, {i['grid_kombinasi']} kombinasi parameter.
Waktu proses {r['waktu_detik']} detik.</p>

<h2>Data</h2>
<p>Jumlah nilai 0: {nol}. Baris lengkap pada empat variabel: {i['baris_lengkap_4_variabel']}.</p>

<h2>Performa tiap skenario</h2>
{_table(r['tabel_performa'], '<b>Tabel 1</b> Performa KNN + Grid Search')}
<p class="note">Selisih dan p dibandingkan dengan Tanpa penanganan (corrected resampled t-test).
Hapus baris dihitung pada data yang berbeda sehingga tidak dibandingkan.
p recall terhadap Tanpa penanganan: {", ".join(f"{k} {_fmt(v)}" for k, v in r['p_recall_vs_tanpa'].items())}.</p>
<p><img src="grafik_akurasi.png" alt="Akurasi nested CV per skenario dengan selang kepercayaan 95%"></p>

<h2>Grid Search</h2>
{_table(r['tabel_grid_search'], '<b>Tabel 2</b> Perubahan parameter dan efek Grid Search')}

<h2>Galat imputasi</h2>
{_table(r['tabel_rmse_imputasi'], '<b>Tabel 3</b> RMSE pengisian ulang nilai yang disembunyikan (makin kecil makin baik)')}
<p class="note">Dihitung pada {r['baris_lengkap_rmse']} baris lengkap, 20% nilai empat variabel disembunyikan, skala terstandardisasi.</p>

<h2>Kenaikan semu (tidak valid)</h2>
<p class="bad">Median per kelas Outcome sebelum pembagian data: {_num(leak['akurasi'], 2)}%
({'+' if leak['selisih_pp'] >= 0 else ''}{_num(leak['selisih_pp'], 2)} pp). Tidak valid karena memakai label data uji.</p>

<h2>Berkas</h2>
<p><a href="hasil.json">hasil.json</a> · <a href="tabel_performa.csv">tabel_performa.csv</a> ·
<a href="tabel_grid_search.csv">tabel_grid_search.csv</a> · <a href="tabel_rmse_imputasi.csv">tabel_rmse_imputasi.csv</a> ·
<a href="akurasi_per_lipatan.csv">akurasi_per_lipatan.csv</a> · <a href="progres.txt">progres.txt</a></p>
"""
    _write(out, "Hasil Eksperimen PKA", body)


def write_status(out):
    log = ""
    p = os.path.join(out, "progres.txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            log = html.escape(f.read())
    body = f"""
<h1>Eksperimen sedang berjalan</h1>
<p class="mut">Halaman ini memuat ulang otomatis setiap 10 detik. Proses penuh memakan waktu sekitar 10-30 menit tergantung CPU.</p>
<pre>{log or 'Menyiapkan...'}</pre>"""
    _write(out, "Eksperimen berjalan", body, refresh=True)


def _write(out, title, body, refresh=False):
    meta = '<meta http-equiv="refresh" content="10">' if refresh else ""
    doc = f"""<!DOCTYPE html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">{meta}
<title>{html.escape(title)}</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>"""
    tmp = os.path.join(out, "index.html.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(doc)
    os.replace(tmp, os.path.join(out, "index.html"))
