"use client";

import { useState } from "react";

type ForecastRow = {
  date: string;
  predicted_flowrate: number;
  precipitation?: number | null;
  et0?: number | null;
};

type ForecastResponse = {
  site: string;
  model: string;
  horizon_days: number;
  historical_average?: number;
  forecast_average?: number;
  forecast: ForecastRow[];
};

const siteOptions = [
  { value: "b", label: "Site B" },
  { value: "k", label: "Site K" },
  { value: "kl", label: "Site KL" },
  { value: "l", label: "Site L" },
  { value: "m", label: "Site M" },
  { value: "s", label: "Site S" },
  { value: "t", label: "Site T" }
];

export default function HomePage() {
  const [site, setSite] = useState("b");
  const [model, setModel] = useState("baseline");
  const [horizonDays, setHorizonDays] = useState(30);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<ForecastResponse | null>(null);

  async function runForecast() {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("/api/forecast", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          site,
          model,
          horizon_days: horizonDays
        })
      });

      const body = await response.json();

      if (!response.ok) {
        throw new Error(
          body.detail || "Forecast request failed."
        );
      }

      setResult(body);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An unknown error occurred."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <header className="header">
        <h1>Water Source Flowrate Forecasting</h1>
        <p>
          Select a site and forecasting period to generate
          the projection.
        </p>
      </header>

      <div className="container">
        <section className="panel">
          <h2>Forecast settings</h2>

          <div className="formGrid">
            <div className="field">
              <label htmlFor="site">Site</label>

              <select
                id="site"
                value={site}
                onChange={(event) =>
                  setSite(event.target.value)
                }
              >
                {siteOptions.map((option) => (
                  <option
                    key={option.value}
                    value={option.value}
                  >
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="field">
              <label htmlFor="model">
                Forecasting model
              </label>

              <select
                id="model"
                value={model}
                onChange={(event) =>
                  setModel(event.target.value)
                }
              >
                <option value="baseline">
                  Baseline
                </option>
                <option value="xgboost">
                  XGBoost
                </option>
                <option value="statistical">
                  Statistical
                </option>
              </select>
            </div>

            <div className="field">
              <label htmlFor="horizon">
                Forecast horizon
              </label>

              <input
                id="horizon"
                type="number"
                min="1"
                max="365"
                value={horizonDays}
                onChange={(event) =>
                  setHorizonDays(
                    Number(event.target.value)
                  )
                }
              />
            </div>
          </div>

          <button
            className="primaryButton"
            onClick={runForecast}
            disabled={loading}
          >
            {loading
              ? "Processing..."
              : "Run forecast"}
          </button>

          {error && (
            <div className="status error">
              {error}
            </div>
          )}
        </section>

        {result && (
          <>
            <section className="panel">
              <h2>Forecast summary</h2>

              <div className="metrics">
                <div className="metric">
                  <h3>Site</h3>
                  <strong>
                    {result.site.toUpperCase()}
                  </strong>
                </div>

                <div className="metric">
                  <h3>Forecast horizon</h3>
                  <strong>
                    {result.horizon_days} days
                  </strong>
                </div>

                <div className="metric">
                  <h3>Forecast average</h3>
                  <strong>
                    {result.forecast_average !== undefined
                      ? result.forecast_average.toFixed(3)
                      : "-"}
                  </strong>
                </div>
              </div>
            </section>

            <section className="panel">
              <h2>Forecast table</h2>

              <div className="tableWrapper">
                <table>
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Predicted flowrate</th>
                      <th>Precipitation</th>
                      <th>ET0</th>
                    </tr>
                  </thead>

                  <tbody>
                    {result.forecast.map((row) => (
                      <tr key={row.date}>
                        <td>{row.date}</td>
                        <td>
                          {row.predicted_flowrate.toFixed(3)}
                        </td>
                        <td>
                          {row.precipitation ?? "-"}
                        </td>
                        <td>{row.et0 ?? "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}
      </div>
    </main>
  );
}
