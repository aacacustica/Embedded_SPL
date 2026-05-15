#!/bin/sh

SENSOR_NAME=$(hostname)
SERVER_BASE="http://100.120.228.23:8080/upload"
PROXY="http://127.0.0.1:1055"

send_file() {
    f="$1"
    subpath="$2"
    base="$(basename "$f")"
    server="${SERVER_BASE}/${subpath}"

    python3 - "$f" "$base" "$server" "$PROXY" <<'PY'
import sys
import urllib.request

fpath, base, server, proxy = sys.argv[1:]
url = f"{server}?name={base}"

with open(fpath, "rb") as fh:
    data = fh.read()

req = urllib.request.Request(url, data=data, method="POST")

# Añadir header mas fiable
req.add_header("X-Filename", base)

opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": proxy, "https": proxy})
)

with opener.open(req, timeout=60) as resp:
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}")
PY
}

send_dir() {
    DIR="$1"
    SUBPATH="$2"

    for f in "$DIR"/*.csv
    do
        [ -f "$f" ] || continue

        base=$(basename "$f")

        # Extraer fecha automáticamente 
        fecha=$(echo "$base" | grep -oE '[0-9]{8}' | head -n1)

        destino="/${SENSOR_NAME}/${SUBPATH}/${fecha}"

        if send_file "$f" "$destino"; then
            rm "$f"
        else
            echo "Error sending $f" >&2
        fi
    done
}

send_dir "/home/pi/data/NOISEPORT-TENERIFE/3-Medidas/P1_CONTENEDORES/AUDIOMOTH/acoustic_params" "acoustics"
send_dir "/home/pi/data/NOISEPORT-TENERIFE/3-Medidas/P1_CONTENEDORES/AUDIOMOTH/predictions_litle" "predictions"

# Opcional: envío de un wav aleatorio
#wav_random=$(find /root/data/NOISEPORT-TENERIFE/3-Medidas/P1_CONTENEDORES/AUDIOMOTH/wav_files -type f -name '*.wav' | shuf -n 1)
#[ -n "$wav_random" ] && send_file "$wav_random" "wav"