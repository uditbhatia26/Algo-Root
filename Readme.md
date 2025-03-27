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

### Screenshots
![image](https://github.com/user-attachments/assets/c6646bbb-136c-42f8-aa76-3279cd7bc32d)
![image](https://github.com/user-attachments/assets/bebfb039-97c7-476e-afeb-a0e239b13e0c)
![image](https://github.com/user-attachments/assets/53da0049-c078-4a62-9208-68824c1e8b3e)


