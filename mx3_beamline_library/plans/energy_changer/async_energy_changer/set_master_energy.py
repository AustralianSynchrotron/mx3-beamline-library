from os import environ
from time import sleep

environ["EPICS_CA_ADDR_LIST"] = "0.0.0.0:5064"

from epics import caget, caput

print(caget("master_energy:master_energy"))
print(caput("master_energy:master_energy", 12))
while caget("master_energy:moving"):
    print("moving...")
    sleep(0.1)
print(caget("master_energy:master_energy"))
