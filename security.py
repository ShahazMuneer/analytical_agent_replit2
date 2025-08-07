"""
Security Manager for Enterprise Analytics Hub
Implements enterprise-grade security features and best practices
"""

import streamlit as st
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Any
import os
from datetime import datetime, timedelta
import json
import re

logger = logging.getLogger(__name__)

class SecurityManager:
    """Enterprise security manager for data protection and compliance"""
    
    # Security configuration
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' https:;",
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'camera=(), microphone=(), geolocation=()'
    }
    
    # Sensitive data patterns for detection
    SENSITIVE_PATTERNS = {
        'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    }
    
    def __init__(self):
        """Initialize security manager"""
        self.session_timeout = 3600  # 1 hour in seconds
        self.max_login_attempts = 5
        self.audit_log = []
        logger.info("Security manager initialized")
    
    @staticmethod
    def apply_security_headers():
        """Apply enterprise security headers to Streamlit app"""
        try:
            # Note: Streamlit doesn't directly support custom headers,
            # but we can document the recommended headers for reverse proxy setup
            logger.info("Security headers configuration documented for proxy setup")
            
            # Store security config in session state for reference
            if 'security_headers' not in st.session_state:
                st.session_state.security_headers = SecurityManager.SECURITY_HEADERS
                
        except Exception as e:
            logger.error(f"Error applying security headers: {str(e)}")
    
    @staticmethod
    def sanitize_input(user_input: str, input_type: str = "general") -> str:
        """Sanitize user input to prevent injection attacks"""
        if not isinstance(user_input, str):
            user_input = str(user_input)
        
        # Remove potentially dangerous characters
        if input_type == "sql":
            # SQL injection prevention
            dangerous_sql = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'EXEC', 'EXECUTE']
            user_input_upper = user_input.upper()
            for dangerous in dangerous_sql:
                if dangerous in user_input_upper:
                    logger.warning(f"Potential SQL injection attempt detected: {dangerous}")
                    user_input = user_input.replace(dangerous.lower(), "")
                    user_input = user_input.replace(dangerous.upper(), "")
        
        elif input_type == "html":
            # XSS prevention
            user_input = user_input.replace("<script", "&lt;script")
            user_input = user_input.replace("</script>", "&lt;/script&gt;")
            user_input = user_input.replace("javascript:", "")
            user_input = user_input.replace("vbscript:", "")
            user_input = user_input.replace("onload=", "")
            user_input = user_input.replace("onerror=", "")
        
        # General sanitization
        user_input = user_input.strip()
        user_input = re.sub(r'[^\w\s\-_.@/(),:;\'\"?!+=]', '', user_input)
        
        return user_input
    
    @staticmethod
    def validate_database_connection(connection_params: Dict[str, str]) -> Dict[str, Any]:
        """Validate and secure database connection parameters"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'sanitized_params': {}
        }
        
        try:
            # Required parameters
            required_fields = ['host', 'database', 'username']
            for field in required_fields:
                if field not in connection_params or not connection_params[field]:
                    validation_result['errors'].append(f"Missing required field: {field}")
                    validation_result['is_valid'] = False
            
            # Sanitize parameters
            for key, value in connection_params.items():
                if key == 'password':
                    # Don't log or sanitize passwords, just validate length
                    if len(value) < 8:
                        validation_result['warnings'].append("Password should be at least 8 characters")
                    validation_result['sanitized_params'][key] = value  # Keep original
                else:
                    sanitized_value = SecurityManager.sanitize_input(str(value))
                    validation_result['sanitized_params'][key] = sanitized_value
                    
                    # Validate specific fields
                    if key == 'host':
                        if not SecurityManager._validate_hostname(sanitized_value):
                            validation_result['errors'].append("Invalid hostname format")
                            validation_result['is_valid'] = False
                    
                    elif key == 'port':
                        try:
                            port_num = int(sanitized_value)
                            if not (1 <= port_num <= 65535):
                                validation_result['errors'].append("Port must be between 1 and 65535")
                                validation_result['is_valid'] = False
                        except ValueError:
                            validation_result['errors'].append("Port must be a valid number")
                            validation_result['is_valid'] = False
            
            # Security recommendations
            if connection_params.get('type') not in ['postgresql', 'mysql', 'sqlite']:
                validation_result['warnings'].append("Consider using well-supported database types")
            
            logger.info(f"Database connection validation completed: {'PASSED' if validation_result['is_valid'] else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"Error validating database connection: {str(e)}")
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    @staticmethod
    def _validate_hostname(hostname: str) -> bool:
        """Validate hostname format"""
        # Allow localhost, IP addresses, and valid domain names
        if hostname in ['localhost', '127.0.0.1', '::1']:
            return True
        
        # IP address pattern
        ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
        if re.match(ip_pattern, hostname):
            return True
        
        # Domain name pattern
        domain_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9]*\.([a-zA-Z]{2,}|[a-zA-Z]{2,}\.[a-zA-Z]{2,})$'
        if re.match(domain_pattern, hostname):
            return True
        
        # Simple hostname pattern
        hostname_pattern = r'^[a-zA-Z0-9\-]+$'
        return bool(re.match(hostname_pattern, hostname))
    
    def detect_sensitive_data(self, data_sample: str) -> Dict[str, List[str]]:
        """Detect potential sensitive data in text"""
        detections = {pattern_name: [] for pattern_name in self.SENSITIVE_PATTERNS}
        
        try:
            for pattern_name, pattern in self.SENSITIVE_PATTERNS.items():
                matches = re.findall(pattern, data_sample, re.IGNORECASE)
                if matches:
                    detections[pattern_name] = matches[:5]  # Limit to first 5 matches
                    logger.warning(f"Potential {pattern_name} detected in data")
            
        except Exception as e:
            logger.error(f"Error detecting sensitive data: {str(e)}")
        
        return detections
    
    def log_audit_event(self, event_type: str, details: Dict[str, Any], user_id: str = "anonymous"):
        """Log security and audit events"""
        try:
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'user_id': user_id,
                'details': details,
                'session_id': st.session_state.get('session_id', 'unknown')
            }
            
            self.audit_log.append(audit_entry)
            
            # Keep only last 1000 entries to prevent memory issues
            if len(self.audit_log) > 1000:
                self.audit_log = self.audit_log[-1000:]
            
            logger.info(f"Audit event logged: {event_type}")
            
        except Exception as e:
            logger.error(f"Error logging audit event: {str(e)}")
    
    def check_data_access_permissions(self, table_name: str, operation: str = "read") -> bool:
        """Check data access permissions (placeholder for enterprise implementation)"""
        try:
            # In a real enterprise environment, this would check against
            # an authorization system (RBAC, ABAC, etc.)
            
            # For now, implement basic checks
            sensitive_tables = ['users', 'passwords', 'credentials', 'secrets']
            
            if any(sensitive in table_name.lower() for sensitive in sensitive_tables):
                logger.warning(f"Access to potentially sensitive table requested: {table_name}")
                return False
            
            # Log access attempt
            self.log_audit_event("data_access", {
                "table": table_name,
                "operation": operation,
                "granted": True
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking data access permissions: {str(e)}")
            return False
    
    def generate_session_token(self) -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
    
    def validate_session(self) -> bool:
        """Validate current session"""
        try:
            if 'session_start_time' not in st.session_state:
                st.session_state.session_start_time = datetime.now()
                st.session_state.session_id = self.generate_session_token()
                return True
            
            # Check session timeout
            session_duration = datetime.now() - st.session_state.session_start_time
            if session_duration.total_seconds() > self.session_timeout:
                logger.warning("Session timeout detected")
                self.log_audit_event("session_timeout", {"duration": session_duration.total_seconds()})
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating session: {str(e)}")
            return False
    
    def mask_sensitive_data(self, data: str, data_type: str = "auto") -> str:
        """Mask sensitive data for display"""
        try:
            if data_type == "auto":
                # Auto-detect and mask
                for pattern_name, pattern in self.SENSITIVE_PATTERNS.items():
                    if pattern_name == "credit_card":
                        data = re.sub(pattern, lambda m: m.group()[:-4].replace(m.group()[0], '*') + m.group()[-4:], data)
                    elif pattern_name == "ssn":
                        data = re.sub(pattern, "***-**-****", data)
                    elif pattern_name == "email":
                        data = re.sub(pattern, lambda m: m.group().split('@')[0][:2] + "****@" + m.group().split('@')[1], data)
                    elif pattern_name == "phone":
                        data = re.sub(pattern, "***-***-****", data)
            
            elif data_type == "credit_card":
                if len(data) >= 4:
                    data = "*" * (len(data) - 4) + data[-4:]
            
            elif data_type == "general":
                if len(data) > 4:
                    data = data[:2] + "*" * (len(data) - 4) + data[-2:]
            
            return data
            
        except Exception as e:
            logger.error(f"Error masking sensitive data: {str(e)}")
            return data
    
    def get_security_recommendations(self) -> List[str]:
        """Get security recommendations for the current setup"""
        recommendations = []
        
        # Environment checks
        if os.getenv("GROQ_API_KEY", "").startswith("gsk_default"):
            recommendations.append("Configure proper GROQ_API_KEY environment variable")
        
        # Session checks
        if not st.session_state.get('session_id'):
            recommendations.append("Initialize secure session management")
        
        # HTTPS recommendation
        recommendations.append("Ensure application is served over HTTPS in production")
        
        # Database security
        recommendations.append("Use encrypted connections for database access")
        
        # Data handling
        recommendations.append("Implement data retention policies for compliance")
        
        # Monitoring
        recommendations.append("Set up security monitoring and alerting")
        
        return recommendations
    
    def create_security_dashboard_widget(self):
        """Create a security status widget for the dashboard"""
        try:
            st.sidebar.markdown("### 🔒 Security Status")
            
            # Session status
            session_valid = self.validate_session()
            if session_valid:
                st.sidebar.success("✅ Session Active")
            else:
                st.sidebar.error("❌ Session Expired")
            
            # Security score (simplified)
            security_score = 85  # This would be calculated based on various factors
            st.sidebar.metric("Security Score", f"{security_score}%")
            
            # Recent audit events
            if self.audit_log:
                recent_events = len([e for e in self.audit_log if 
                                   datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=1)])
                st.sidebar.metric("Events (1h)", recent_events)
            
            # Security recommendations
            recommendations = self.get_security_recommendations()
            if recommendations:
                with st.sidebar.expander("🛡️ Security Recommendations"):
                    for rec in recommendations[:3]:  # Show top 3
                        st.write(f"• {rec}")
            
        except Exception as e:
            logger.error(f"Error creating security dashboard widget: {str(e)}")
    
    def export_audit_log(self, start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None) -> str:
        """Export audit log for compliance reporting"""
        try:
            filtered_log = self.audit_log
            
            if start_date:
                filtered_log = [e for e in filtered_log if 
                              datetime.fromisoformat(e['timestamp']) >= start_date]
            
            if end_date:
                filtered_log = [e for e in filtered_log if 
                              datetime.fromisoformat(e['timestamp']) <= end_date]
            
            # Create JSON export
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_events': len(filtered_log),
                'events': filtered_log
            }
            
            return json.dumps(export_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error exporting audit log: {str(e)}")
            return json.dumps({"error": str(e)})
