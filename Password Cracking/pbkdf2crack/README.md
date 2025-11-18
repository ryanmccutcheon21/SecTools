# 🔐 PBKDF2 Cracking Tool (Python)

A multi-threaded, rule-mangling PBKDF2 password cracker written in Python.

Supports:
- ✅ Full hash parsing (e.g., `pbkdf2:sha256:600000$salt$hash`)
- ✅ Salt/hash/iterations as arguments
- ✅ SHA1, SHA256, SHA512
- ✅ Hashcat-style basic rules
- ✅ Multiprocessing
- ✅ Progress bar, ETA, and hash speed
- ✅ Colored output (optional)

---

## 🚀 Usage

### 📦 Install requirements

```
pip install tqdm colorama
```

🔧 Basic usage

```
python3 crack_pbkdf2.py 'pbkdf2:sha256:600000$salt$hash'
```

🧂 Manual mode

```
python3 crack_pbkdf2.py -s saltval -H targethash -i 600000 -a sha256 -w rockyou.txt
```

⚙️ Options

```
Option	Shorthand	Description
--salt	-s	Salt string
--hash	-H	Hash (hex)
--iterations	-i	Iteration count
--algorithm	-a	sha1, sha256, or sha512
--wordlist	-w	Path to .txt or .gz wordlist
--no-color		Disable colored output
--verbose	-v	More output
--quiet	-q	Minimal output
```

📜 License
MIT License – see LICENSE file.

---

### ✅ `LICENSE` (MIT License)

```text
MIT License

Copyright (c) 2025 Gizmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```
