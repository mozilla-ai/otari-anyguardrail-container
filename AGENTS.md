# General guidelines

For all these guidelines, provide a clear indication with a warning sign (⚠️) and a short description in case some exception needs to be made.

* Use pytest instead of unittest
* Always use async if possible
* Always use an async test client for http tests if possible
* ty is used to check types, use it too whenever tests would be run
* In the case of uncertainty or excessive complexity, stop and ask, suggesting alternatives if appropriate