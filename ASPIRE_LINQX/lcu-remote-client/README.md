# LCU Remote Client Library

This library functions as a proxy for interfacing with a remote LCU. Currently, the method to register an item is implemented.
It can be installed with `pip install dist/LCUClient-0.1.0-py3-none-any.whl`.

It can be called like so:
```
from LCUClient import client
c = client.LCUClient("34.136.227.69:8000")
id = "9912-30042" # identifier string
description = "Mass Spectrometer"
result = c.register(id, description)
print(c.status())
```

The response is a message indicating successful invocation of the remote method.