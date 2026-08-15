package llm

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	// PlatformDefaultName is the Settings value for the deploy-time env connection.
	PlatformDefaultName = "platform_default"

	envBaseURL = "OPENVEEVA_LLM_BASE_URL"
	envAPIKey  = "OPENVEEVA_LLM_API_KEY"
	envModel   = "OPENVEEVA_LLM_MODEL"
)

// Resolver loads LLM connection facts (env platform_default and optional vault rows).
type Resolver struct {
	pool *pgxpool.Pool
}

// NewResolver returns a connection resolver. pool may be nil (env-only).
func NewResolver(pool *pgxpool.Pool) *Resolver {
	return &Resolver{pool: pool}
}

// Resolve returns a connection by name. Empty name or platform_default uses env.
func (r *Resolver) Resolve(ctx context.Context, vaultID uuid.UUID, name string) (Connection, error) {
	name = strings.TrimSpace(name)
	if name == "" || name == PlatformDefaultName {
		return platformDefaultFromEnv()
	}
	if r == nil || r.pool == nil {
		return Connection{}, fmt.Errorf("llm connection %q not found (no connection store)", name)
	}
	var conn Connection
	var active bool
	err := r.pool.QueryRow(ctx, `
		SELECT name, base_url, api_key, model, active
		FROM zint_llm_connection
		WHERE vault_id = $1 AND name = $2`,
		vaultID, name,
	).Scan(&conn.Name, &conn.BaseURL, &conn.APIKey, &conn.Model, &active)
	if err != nil {
		if err == pgx.ErrNoRows {
			return Connection{}, fmt.Errorf("llm connection %q not found", name)
		}
		return Connection{}, fmt.Errorf("lookup llm connection: %w", err)
	}
	if !active {
		return Connection{}, fmt.Errorf("llm connection %q is inactive", name)
	}
	return conn, nil
}

func platformDefaultFromEnv() (Connection, error) {
	base := strings.TrimSpace(os.Getenv(envBaseURL))
	key := strings.TrimSpace(os.Getenv(envAPIKey))
	model := strings.TrimSpace(os.Getenv(envModel))
	if base == "" || key == "" || model == "" {
		return Connection{}, fmt.Errorf("platform_default LLM requires %s, %s, and %s", envBaseURL, envAPIKey, envModel)
	}
	return Connection{
		Name:    PlatformDefaultName,
		BaseURL: base,
		APIKey:  key,
		Model:   model,
	}, nil
}

// UpsertConnection inserts or updates a vault-scoped LLM connection (api_key stored as provided).
func (r *Resolver) UpsertConnection(ctx context.Context, vaultID uuid.UUID, conn Connection, active bool) error {
	if r == nil || r.pool == nil {
		return fmt.Errorf("llm connection store not configured")
	}
	conn.Name = strings.TrimSpace(conn.Name)
	conn.BaseURL = strings.TrimSpace(conn.BaseURL)
	conn.APIKey = strings.TrimSpace(conn.APIKey)
	conn.Model = strings.TrimSpace(conn.Model)
	if conn.Name == "" || conn.Name == PlatformDefaultName {
		return fmt.Errorf("invalid connection name")
	}
	if conn.BaseURL == "" || conn.APIKey == "" || conn.Model == "" {
		return fmt.Errorf("base_url, api_key, and model required")
	}
	_, err := r.pool.Exec(ctx, `
		INSERT INTO zint_llm_connection (vault_id, name, base_url, api_key, model, active, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, clock_timestamp())
		ON CONFLICT (vault_id, name) DO UPDATE SET
			base_url = EXCLUDED.base_url,
			api_key = EXCLUDED.api_key,
			model = EXCLUDED.model,
			active = EXCLUDED.active,
			updated_at = clock_timestamp()`,
		vaultID, conn.Name, conn.BaseURL, conn.APIKey, conn.Model, active,
	)
	if err != nil {
		return fmt.Errorf("upsert llm connection: %w", err)
	}
	return nil
}
