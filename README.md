# AutoML

## Frontend Migration to Next.js (Flask Deprecated)

The legacy Flask UI served from `templates/` and `static/` is now deprecated. The project uses the Next.js app in `UI/` for all user-facing functionality. The Python side continues to provide AutoML logic via `runner.py`, invoked from the Next.js API.

### Development (Windows PowerShell)

- Prerequisites:
	- Python 3.10+ available on PATH (`python` command)
	- Node.js 18+
	- pnpm installed (`npm i -g pnpm`)

- Start Next.js UI:

```powershell
Push-Location "c:\Users\sarve\Desktop\BTP\AutoML\UI"; pnpm install; pnpm dev
```

- Upload API endpoint:
	- Next.js API route: `POST http://localhost:3000/api/upload`
	- Form fields:
		- `dataset`: CSV file
		- `problem_type`: `regression` | `classification` | `clustering`
		- `target_col`: required for `regression` and `classification`

During an upload, the file is saved under the workspace `uploaded_files/` folder and `runner.py` is executed with appropriate arguments. Results are returned as JSON to the UI.

### Flask Status

- `app.py` and related Flask files are kept for reference but are not used by the UI. Consider them deprecated.

### Production Options

- Deploy `UI/` to Vercel or a Node server. Ensure Python is available on the server if using the Next API to spawn `runner.py`.
- Alternatively, host Python on a separate service and adapt the Next API route to call that service.

### Troubleshooting

- If the Next API cannot find Python, ensure `python` is resolvable on PATH. You can change the command in `UI/app/api/upload/route.ts`.
- Large files: consider setting platform upload limits and monitoring execution time of `runner.py`.
A pipeline to automate generation of trained ML models
