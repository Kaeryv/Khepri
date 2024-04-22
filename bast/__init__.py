import os
if "BAST_MT_ON" in os.environ:
    BAST_MT_ON=bool(int(os.environ["BAST_MT_ON"]))
else:
    BAST_MT_ON=True
