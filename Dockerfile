# Temel hafif Python imaji
FROM python:3.10-slim

WORKDIR /app

# Gereksinimleri kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Firewall islemleri icin root yetkisine ihtiyac var (iptables).
# Bu yuzden USER degistirmeden root olarak birakiyoruz, ancak
# ayricaliklari docker-compose uzerinden kisitlayacagiz (cap_drop).

# Proje dosyalarini kopyala
COPY . .

# NOT: Bu uygulama iptables uzerinden IP ban islemi yaptiginden dolayi
# NET_ADMIN capability'sine ve root kullaniciya ihtiyac duyar.
# Bu yuzden non-root kullaniciya gecis yapilmiyor.
# Capability kisitlamasi docker-compose.yml tarafinda cap_drop/cap_add ile saglaniyor.

# Saglik kontrolu -- ana Python surecinin ayakta olup olmadigini dogrular
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
  CMD pgrep -f "python main.py" || exit 1

# Konteyner ayaga kalktiginda daemon modunda motoru calistir
CMD ["python", "main.py", "--mode", "daemon"]