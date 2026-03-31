.PHONY: bootstrap test app docker-up docker-down

bootstrap:
	python scripts/bootstrap.py --count 500

test:
	python -m pytest -q

app:
	streamlit run app/streamlit_app.py

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

