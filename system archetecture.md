flowchart TB
    CSV[("load_weather_full.csv\n2019–2022 hourly data")]
    SIM["Simulator\napp/simulator.py\n1 row/sec · 2019-01-01 onwards"]
    CSV --> SIM

    subgraph ports["Port-Forwards (local)"]
        P1["localhost:8001"]
        P2["localhost:8002"]
        P3["localhost:8003"]
        P4["localhost:9090"]
        P5["localhost:3000"]
    end

    SIM --> P1
    SIM --> P2
    SIM --> P3

    subgraph k8s["Kubernetes — namespace: load-forecast"]

        subgraph svc["Inference Services  (same image: load-forecast-api:v10)"]
            STATIC["Static Service\nMODE=static\nBase RF model — never retrains\nDeployment: load-forecast-static"]
            DYNAMIC["Dynamic Service\nMODE=dynamic\nHot-swaps model via /reload\nDeployment: load-forecast-dynamic"]
            INLINE["Inline Service\nMODE=inline\nRetrains in background thread\nDeployment: load-forecast-inline"]
        end

        subgraph models["Model Artifacts"]
            IMG[/"Base model (baked into image)\nrf_2018.pkl + scalers"/]
            PVC[("PVC: model-storage\nrf_retrained.pkl + scalers")]
        end

        DRIFT["Drift Monitor\napp/drift_monitor.py\nPolls Prometheus every 60s\nDeployment: drift-monitor"]

        JOB["Retrain Job\n(k8s Job, spawned on demand)\napp/retrain.py\nTrains RF on last 1 year of data"]

        PROM["Prometheus\n1s scrape interval\nDeployment: prometheus"]
        GRAFANA["Grafana\nProvisioned dashboard\nDeployment: grafana"]

        STATIC --- IMG
        DYNAMIC --- IMG
        DYNAMIC <-.->|retrained model| PVC
        INLINE --- IMG

        STATIC -->|/metrics| PROM
        DYNAMIC -->|/metrics| PROM
        INLINE -->|/metrics| PROM

        PROM -->|rolling_mae_mw\ndrift_active| DRIFT
        DRIFT -->|spawns when MAE > 2000 MW\n14-day cooldown| JOB
        JOB -->|writes rf_retrained.pkl| PVC
        DRIFT -->|POST /reload\nafter job succeeds| DYNAMIC

        INLINE -->|background thread\nwhen MAE > 2000 MW| INLINE

        PROM --> GRAFANA
    end

    P1 --> STATIC
    P2 --> DYNAMIC
    P3 --> INLINE
    P4 --> PROM
    P5 --> GRAFANA
