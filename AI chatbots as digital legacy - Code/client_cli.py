import os, sys, requests

def main():
    base = os.environ.get("LUNIBOT_BASE", "http://localhost:8000")
    message = " ".join(sys.argv[1:]) or "Sag kurz Hi und stell dich vor."
    r = requests.post(f"{base}/chat", json={"message": message})
    r.raise_for_status()
    data = r.json()
    print("\n=== Reply ===\n")
    print(data.get("reply", "<no reply>"))

if __name__ == "__main__":
    main()
