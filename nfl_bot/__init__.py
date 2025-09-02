"""Utilities and domain constants for the NFL bot package."""

TEAM_CODE_TO_FULL = {
    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens', 'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears', 'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys', 'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars', 'KC': 'Kansas City Chiefs',
    'LAC': 'Los Angeles Chargers', 'LAR': 'Los Angeles Rams', 'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings', 'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
    'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers', 'SEA': 'Seattle Seahawks',
    'SF': 'San Francisco 49ers', 'TB': 'Tampa Bay Buccaneers', 'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders',
    'WSH': 'Washington Commanders'
}

FULL_TO_CODE = {v: k for k, v in TEAM_CODE_TO_FULL.items()}

# Stadium coordinates (lat, lon) used for weather lookups.
STADIUM_COORDS = {
    'ARI': (33.5275, -112.2626), 'ATL': (33.7554, -84.4008), 'BAL': (39.2780, -76.6227),
    'BUF': (42.7738, -78.7870), 'CAR': (35.2258, -80.8528), 'CHI': (41.8623, -87.6167),
    'CIN': (39.0954, -84.5161), 'CLE': (41.5061, -81.6995), 'DAL': (32.7473, -97.0945),
    'DEN': (39.7439, -105.0201), 'DET': (42.3400, -83.0456), 'GB': (44.5013, -88.0622),
    'HOU': (29.6847, -95.4107), 'IND': (39.7601, -86.1639), 'JAX': (30.3239, -81.6374),
    'KC': (39.0490, -94.4839), 'LAC': (33.9535, -118.3387), 'LAR': (33.9535, -118.3387),
    'LV': (36.0909, -115.1830), 'MIA': (25.9580, -80.2389), 'MIN': (44.9738, -93.2577),
    'NE': (42.0909, -71.2643), 'NO': (29.9509, -90.0815), 'NYG': (40.8128, -74.0742),
    'NYJ': (40.8128, -74.0742), 'PHI': (39.9008, -75.1675), 'PIT': (40.4468, -80.0158),
    'SEA': (47.5952, -122.3316), 'SF': (37.4030, -121.9700), 'TB': (27.9759, -82.5033),
    'TEN': (36.1665, -86.7713), 'WAS': (38.9078, -76.8645), 'WSH': (38.9078, -76.8645),
}

__all__ = ["TEAM_CODE_TO_FULL", "FULL_TO_CODE", "STADIUM_COORDS"]
