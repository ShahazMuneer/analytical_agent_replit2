# Enterprise Analytics Hub

## Overview

This is an advanced enterprise-grade analytics platform built with Streamlit that provides AI-powered business intelligence capabilities for banking and finance organizations. The platform features natural language to SQL conversion, multi-database connectivity, advanced forecasting, and comprehensive business analytics dashboards.

## Recent Changes (August 1, 2025)

- **Major UI Overhaul**: Transformed into enterprise-grade interface with professional styling and banking/finance focus
- **Multi-Database Support**: Added connectivity for PostgreSQL, MySQL, Teradata, Hive, SQL Server, and Oracle with secure credential management
- **Advanced Analytics Engine**: Implemented comprehensive business intelligence features including KPI dashboards, executive summaries, and multi-chart analytics
- **Enhanced Visualization**: Fixed time series chart generation issues and added intelligent chart selection based on data patterns
- **Results Display Enhancement**: Added pagination, full table display options (10-100 rows), and improved CSV download functionality
- **Forecasting Tab**: Created dedicated advanced forecasting interface with customizable parameters and confidence intervals
- **Step-by-Step Analytics**: Added comprehensive 6-step data analysis workflow with data quality assessment, statistical analysis, pattern detection, and strategic recommendations
- **Welcome Experience**: Created AI-driven intelligence hub welcome screen with feature showcase and quick-start options
- **Teradata Integration**: Enhanced enterprise Teradata connectivity with direct connection and ODBC options, authentication methods, and advanced settings
- **Banking Focus**: Redesigned interface specifically for banking and finance professionals with industry-relevant features and terminology

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

- **Frontend**: Streamlit web application providing the user interface
- **Backend**: Python modules handling database operations, AI integration, forecasting, and visualization
- **Data Layer**: SQLite database with sample business data (products, customers, sales)
- **AI Integration**: Groq API for natural language to SQL conversion
- **Analytics**: Time series forecasting using Facebook Prophet (with fallback to simple trend analysis)

## Key Components

### 1. Main Application (`app.py`)
- **Purpose**: Central Streamlit application orchestrating all components
- **Architecture Decision**: Uses Streamlit's session state for maintaining query history and results
- **Rationale**: Streamlit provides rapid prototyping capabilities with built-in state management

### 2. Database Manager (`database.py`)
- **Purpose**: Handles SQLite database operations and sample data generation
- **Architecture Decision**: Uses SQLite with a denormalized view for analytics
- **Rationale**: SQLite provides zero-configuration setup while the denormalized view simplifies query generation for AI

### 3. Groq Client (`groq_client.py`)
- **Purpose**: Interfaces with Groq API for natural language to SQL conversion
- **Architecture Decision**: Uses Llama 3.1 70B model with structured prompts
- **Rationale**: Groq provides fast inference for large language models, enabling real-time query conversion

### 4. Forecasting Engine (`forecasting.py`)
- **Purpose**: Provides time series forecasting capabilities
- **Architecture Decision**: Primary Prophet implementation with simple trend fallback
- **Rationale**: Prophet handles seasonality and trends well, while fallback ensures functionality when Prophet is unavailable

### 5. Visualization Generator (`visualizations.py`)
- **Purpose**: Creates interactive charts and handles forecast visualization
- **Architecture Decision**: Uses Plotly for interactive visualizations
- **Rationale**: Plotly integrates well with Streamlit and provides rich interactive features

## Data Flow

1. **User Input**: User enters natural language query in Streamlit interface
2. **Query Processing**: Groq client converts natural language to SQL
3. **Data Retrieval**: Database manager executes SQL query against SQLite database
4. **Analysis Detection**: System detects if forecasting or time series visualization is needed
5. **Visualization**: Chart generator creates appropriate visualizations
6. **Forecasting** (if needed): Forecasting engine generates future predictions
7. **Display**: Results, charts, and forecasts are displayed in Streamlit interface

## External Dependencies

### Required APIs
- **Groq API**: For natural language to SQL conversion
- **API Key**: Configured via `GROQ_API_KEY` environment variable

### Python Libraries
- **Core**: `streamlit`, `pandas`, `plotly`
- **Database**: `sqlite3`
- **AI/ML**: `groq`, `prophet` (optional), `numpy`
- **Utilities**: `datetime`, `json`, `re`, `os`, `random`

### Forecasting Dependencies
- **Primary**: Facebook Prophet for advanced time series forecasting
- **Fallback**: Simple trend analysis using numpy when Prophet is unavailable

## Deployment Strategy

### Local Development
- **Database**: SQLite file (`analytics.db`) created automatically
- **Initialization**: `init_db.py` script for database setup
- **Environment**: Requires `GROQ_API_KEY` environment variable

### Production Considerations
- **Database**: Current SQLite setup suitable for demo/development
- **Scaling**: Would require migration to PostgreSQL/MySQL for production
- **API Keys**: Secure environment variable management needed
- **Dependencies**: Prophet installation may require additional system dependencies

### Architecture Benefits
- **Modularity**: Each component can be developed and tested independently
- **Flexibility**: Easy to swap out components (e.g., different AI providers)
- **Extensibility**: New visualization types and forecasting methods can be added easily
- **User-Friendly**: Natural language interface removes SQL knowledge barrier

### Trade-offs
- **AI Dependency**: Requires external API for core functionality
- **Database Limitations**: SQLite limits concurrent access and data size
- **Forecasting Complexity**: Prophet dependency may complicate deployment
- **Error Handling**: Limited fallback options if AI service is unavailable