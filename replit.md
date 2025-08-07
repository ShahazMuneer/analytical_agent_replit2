# Enterprise Analytics Hub

## Overview

The Enterprise Analytics Hub is an AI-powered business intelligence platform designed specifically for banking and finance organizations. Built with Python and Streamlit, it combines natural language query processing with comprehensive data analytics capabilities. The platform enables users to convert plain English questions into SQL queries, perform advanced forecasting, and generate professional visualizations. It supports multiple database connections, provides executive dashboards, and includes enterprise-grade security features for financial institutions.

## User Preferences

Preferred communication style: Simple, everyday language.
Interface preference: Single-page comprehensive minimal design (as of Aug 6, 2025)
Database display requirement: Show multiple database connectivity options prominently
Tool preference: Avoid PyGWalker and complex data profiling tools

## System Architecture

### Frontend Architecture
The application uses **Streamlit** as the web framework, providing an interactive dashboard interface. The UI is organized into multiple tabs including AI Query, Forecasting, Step-by-Step Analytics, and Data Profiling. The design uses professional styling with enterprise color schemes suitable for banking environments. Session state management maintains query history and current results across user interactions.

### Backend Architecture
The system follows a **modular component architecture** with specialized managers:
- **DatabaseManager**: Handles database connections and operations across multiple database types (SQLite, PostgreSQL, MySQL, Oracle, SQL Server, Teradata, Hive)
- **GroqClient**: Manages AI integration for natural language to SQL conversion using Llama 3.1 70B model
- **ForecastingEngine**: Provides time series forecasting using Facebook Prophet with statistical fallbacks
- **ChartGenerator**: Creates intelligent visualizations with automatic chart type selection
- **SecurityManager**: Implements enterprise security headers, audit logging, and compliance features

### Data Storage Solutions
The primary data layer uses **SQLite** for local development with a denormalized view (`sales_product_customer_view`) that combines sales, product, and customer data. This design simplifies AI query generation by providing a single queryable surface. The system supports external database connections through SQLAlchemy, enabling enterprise deployments with existing data infrastructure.

### Authentication and Authorization
Enterprise security is implemented through the **SecurityManager** component, which applies bank-grade security headers, implements session timeouts, tracks login attempts, and provides audit logging capabilities. The system includes data privacy features with sensitive data pattern detection for PII protection.

### Analytics Engine
The **StepByStepAnalytics** component provides a comprehensive 6-step analysis workflow including data quality assessment, statistical analysis, pattern detection, and strategic recommendations. The system automatically suggests appropriate analysis types based on data patterns and user queries.

## External Dependencies

### AI Services
- **Groq API**: Primary natural language processing service using Llama 3.1 70B model for converting English queries to SQL. Requires GROQ_API_KEY environment variable.

### Database Connectivity
- **SQLAlchemy**: Database abstraction layer supporting multiple database engines
- **psycopg2**: PostgreSQL adapter
- **pymysql**: MySQL connector
- **pyodbc**: ODBC driver for SQL Server and other databases
- **teradatasql**: Native Teradata connector
- **pyhive**: Apache Hive connectivity

### Analytics and Visualization
- **Facebook Prophet**: Time series forecasting library with seasonal decomposition
- **Plotly**: Interactive visualization library for charts and dashboards
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing foundation

### Data Profiling Tools
- **PyGWalker**: Tableau-style visual analysis tool
- **D-Tale**: Interactive data exploration interface
- **ydata-profiling**: Automated data profiling reports
- **SweetViz**: Comparative data analysis

### Web Framework
- **Streamlit**: Primary web application framework providing the user interface and session management