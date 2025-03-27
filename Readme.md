# Flask Automation API 🚀

A simple Flask API that generates executable Python code for automation tasks.

## How to Use

1. **Run the API**:

   ```bash
   python app.py
   ```

2. **Send a Request**:

   ```bash
   curl -X POST http://127.0.0.1:5000/execute -H "Content-Type: application/json" -d '{"prompt": "Open calculator"}'
   ```

3. **Response Example**:
   ```json
   {
     "function": "open_calculator",
     "code": "import os\nos.system('calc' if os.name == 'nt' else 'gnome-calculator')"
   }
   ```
