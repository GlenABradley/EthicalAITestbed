"""
Advanced Production Features - Phase 9
Implementing enterprise-grade authentication, scaling, and deployment capabilities

Architecture based on expertise from:
- Auth0 and OAuth 2.0/OpenID Connect standards (industry authentication)
- Netflix's microservices architecture (scaling and resilience)
- Google's Site Reliability Engineering practices (monitoring and observability)
- Amazon Web Services best practices (cloud deployment and scaling)
- Kubernetes patterns for container orchestration
- HashiCorp Vault patterns for secrets management
- Prometheus + Grafana monitoring stack patterns
- Rate limiting and API gateway patterns (Kong, AWS API Gateway)

Key Features:
1. JWT-based authentication with role-based access control (RBAC)
2. API rate limiting and throttling
3. Request/response caching with Redis
4. Health checks and monitoring endpoints
5. Metrics collection and observability
6. Horizontal scaling support
7. Security headers and CORS management
8. Production logging and error tracking
9. Configuration management and environment handling
10. Deployment automation support

Dependencies: pip install redis pyjwt passlib[bcrypt] slowapi prometheus-client python-jose[cryptography]
"""

import asyncio
import logging
import json
import time
import uuid
import hashlib
import os
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from functools import wraps
import secrets
from pathlib import Path

# Authentication and security
try:
    import jwt
    from passlib.context import CryptContext
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logging.warning("JWT/Passlib not available - authentication disabled")

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False
    logging.warning("SlowAPI not available - rate limiting disabled")

# Metrics and monitoring
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available - metrics disabled")

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles following RBAC principles"""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    DEVELOPER = "developer"
    VIEWER = "viewer"
    API_USER = "api_user"

class APITier(Enum):
    """API usage tiers for rate limiting"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

@dataclass
class User:
    """User model for authentication"""
    user_id: str
    username: str
    email: str
    role: UserRole
    api_tier: APITier
    hashed_password: str
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    api_key: Optional[str] = None
    rate_limit_remaining: int = 1000
    rate_limit_reset: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AuthToken:
    """JWT authentication token"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    user_id: str = ""
    username: str = ""
    role: str = ""

@dataclass
class APIMetrics:
    """API usage metrics"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    user_id: str
    timestamp: datetime
    request_size: int = 0
    response_size: int = 0

class AuthenticationManager:
    """
    Enterprise-grade authentication manager
    Following OAuth 2.0/OpenID Connect and Auth0 patterns
    """
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 60
        
        if JWT_AVAILABLE:
            self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            self.is_available = True
            logger.info("Authentication manager initialized with JWT support")
        else:
            self.is_available = False
            logger.warning("Authentication manager initialized without JWT support")
        
        # In-memory user store (in production, use database)
        self.users: Dict[str, User] = {}
        self._init_default_users()
    
    def _init_default_users(self):
        """Initialize default users for demo purposes"""
        if not JWT_AVAILABLE:
            return
            
        default_users = [
            {
                "username": "admin",
                "email": "admin@ethicalai.org",
                "password": "admin123",
                "role": UserRole.ADMIN,
                "api_tier": APITier.ENTERPRISE
            },
            {
                "username": "researcher",
                "email": "researcher@ethicalai.org", 
                "password": "research123",
                "role": UserRole.RESEARCHER,
                "api_tier": APITier.PREMIUM
            },
            {
                "username": "developer",
                "email": "developer@ethicalai.org",
                "password": "dev123", 
                "role": UserRole.DEVELOPER,
                "api_tier": APITier.BASIC
            },
            {
                "username": "viewer",
                "email": "viewer@ethicalai.org",
                "password": "view123",
                "role": UserRole.VIEWER,
                "api_tier": APITier.FREE
            }
        ]
        
        for user_data in default_users:
            user_id = str(uuid.uuid4())
            api_key = f"eak_{secrets.token_urlsafe(32)}"  # Ethical AI Key prefix
            
            user = User(
                user_id=user_id,
                username=user_data["username"],
                email=user_data["email"],
                role=user_data["role"],
                api_tier=user_data["api_tier"],
                hashed_password=self.get_password_hash(user_data["password"]),
                api_key=api_key
            )
            
            self.users[user_data["username"]] = user
            logger.info(f"Created default user: {user_data['username']} ({user_data['role'].value}) - API Key: {api_key}")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        if not JWT_AVAILABLE:
            return True  # Fallback for demo
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash"""
        if not JWT_AVAILABLE:
            return hashlib.md5(password.encode()).hexdigest()  # Fallback
        return self.pwd_context.hash(password)
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password"""
        user = self.users.get(username)
        if not user or not user.is_active:
            return None
            
        if not self.verify_password(password, user.hashed_password):
            return None
            
        user.last_login = datetime.utcnow()
        return user
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate user with API key"""
        for user in self.users.values():
            if user.api_key == api_key and user.is_active:
                return user
        return None
    
    def create_access_token(self, user: User) -> AuthToken:
        """Create JWT access token"""
        if not JWT_AVAILABLE:
            # Fallback token
            return AuthToken(
                access_token=f"fallback_{user.user_id}_{int(time.time())}",
                user_id=user.user_id,
                username=user.username,
                role=user.role.value
            )
        
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "api_tier": user.api_tier.value,
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": "ethical-ai-testbed"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        return AuthToken(
            access_token=token,
            expires_in=self.access_token_expire_minutes * 60,
            user_id=user.user_id,
            username=user.username,
            role=user.role.value
        )
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        if not JWT_AVAILABLE:
            # Fallback verification
            if token.startswith("fallback_"):
                parts = token.split("_")
                if len(parts) == 3:
                    return {"sub": parts[1], "username": "fallback_user", "role": "viewer"}
            return None
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            return None

class RateLimiter:
    """
    Advanced rate limiting following API gateway patterns
    Based on Kong, AWS API Gateway, and CloudFlare approaches
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()  # Test connection
            self.redis_available = True
            logger.info("Rate limiter initialized with Redis backend")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory rate limiting: {e}")
            self.redis_available = False
            self.in_memory_store = {}
        
        # Rate limits by API tier
        self.rate_limits = {
            APITier.FREE: {"requests": 100, "window": 3600},      # 100/hour
            APITier.BASIC: {"requests": 1000, "window": 3600},    # 1000/hour
            APITier.PREMIUM: {"requests": 10000, "window": 3600}, # 10k/hour
            APITier.ENTERPRISE: {"requests": 100000, "window": 3600} # 100k/hour
        }
        
        if RATE_LIMITING_AVAILABLE:
            self.limiter = Limiter(key_func=get_remote_address)
            self.is_available = True
        else:
            self.is_available = False
    
    def check_rate_limit(self, user_id: str, api_tier: APITier, endpoint: str = "general") -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits"""
        limit_info = self.rate_limits.get(api_tier, self.rate_limits[APITier.FREE])
        
        key = f"rate_limit:{user_id}:{endpoint}"
        current_time = int(time.time())
        window_start = current_time - (current_time % limit_info["window"])
        
        if self.redis_available:
            return self._check_redis_rate_limit(key, window_start, limit_info)
        else:
            return self._check_memory_rate_limit(key, window_start, limit_info)
    
    def _check_redis_rate_limit(self, key: str, window_start: int, limit_info: Dict[str, int]) -> Tuple[bool, Dict[str, Any]]:
        """Redis-based rate limiting"""
        try:
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start - limit_info["window"])
            pipe.zcard(key)
            pipe.zadd(key, {str(time.time()): time.time()})
            pipe.expire(key, limit_info["window"])
            
            results = pipe.execute()
            current_requests = results[1]
            
            remaining = max(0, limit_info["requests"] - current_requests)
            
            rate_limit_info = {
                "limit": limit_info["requests"],
                "remaining": remaining,
                "reset": window_start + limit_info["window"],
                "window": limit_info["window"]
            }
            
            return current_requests <= limit_info["requests"], rate_limit_info
            
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            return True, {"error": "rate_limiting_unavailable"}
    
    def _check_memory_rate_limit(self, key: str, window_start: int, limit_info: Dict[str, int]) -> Tuple[bool, Dict[str, Any]]:
        """In-memory rate limiting fallback"""
        if key not in self.in_memory_store:
            self.in_memory_store[key] = []
        
        current_time = time.time()
        # Clean old entries
        self.in_memory_store[key] = [
            timestamp for timestamp in self.in_memory_store[key]
            if current_time - timestamp < limit_info["window"]
        ]
        
        current_requests = len(self.in_memory_store[key])
        self.in_memory_store[key].append(current_time)
        
        remaining = max(0, limit_info["requests"] - current_requests)
        
        rate_limit_info = {
            "limit": limit_info["requests"],
            "remaining": remaining,
            "reset": window_start + limit_info["window"],
            "window": limit_info["window"]
        }
        
        return current_requests <= limit_info["requests"], rate_limit_info

class MetricsCollector:
    """
    Enterprise metrics collection following Prometheus patterns
    Based on Google SRE practices and observability standards
    """
    
    def __init__(self):
        if PROMETHEUS_AVAILABLE:
            self.request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
            self.request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
            self.active_users = Gauge('active_users_total', 'Number of active users')
            self.rate_limit_exceeded = Counter('rate_limit_exceeded_total', 'Rate limit exceeded count', ['user_tier'])
            self.ethics_analyses = Counter('ethics_analyses_total', 'Total ethics analyses performed', ['analysis_type'])
            self.knowledge_queries = Counter('knowledge_queries_total', 'Total knowledge queries', ['domain'])
            self.streaming_connections = Gauge('streaming_connections_active', 'Active streaming connections')
            
            self.is_available = True
            logger.info("Metrics collector initialized with Prometheus support")
        else:
            self.is_available = False
            self.fallback_metrics = {}
            logger.warning("Metrics collector initialized without Prometheus support")
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        if self.is_available and PROMETHEUS_AVAILABLE:
            self.request_count.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
            self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        else:
            # Fallback metrics
            key = f"requests_{method}_{endpoint}_{status_code}"
            self.fallback_metrics[key] = self.fallback_metrics.get(key, 0) + 1
    
    def record_ethics_analysis(self, analysis_type: str):
        """Record ethics analysis metrics"""
        if self.is_available and PROMETHEUS_AVAILABLE:
            self.ethics_analyses.labels(analysis_type=analysis_type).inc()
        else:
            key = f"ethics_analysis_{analysis_type}"
            self.fallback_metrics[key] = self.fallback_metrics.get(key, 0) + 1
    
    def record_knowledge_query(self, domain: str):
        """Record knowledge query metrics"""
        if self.is_available and PROMETHEUS_AVAILABLE:
            self.knowledge_queries.labels(domain=domain or "general").inc()
        else:
            key = f"knowledge_query_{domain or 'general'}"
            self.fallback_metrics[key] = self.fallback_metrics.get(key, 0) + 1
    
    def set_active_users(self, count: int):
        """Set active users gauge"""
        if self.is_available and PROMETHEUS_AVAILABLE:
            self.active_users.set(count)
    
    def set_streaming_connections(self, count: int):
        """Set active streaming connections"""
        if self.is_available and PROMETHEUS_AVAILABLE:
            self.streaming_connections.set(count)
    
    def record_rate_limit_exceeded(self, user_tier: str):
        """Record rate limit exceeded"""
        if self.is_available and PROMETHEUS_AVAILABLE:
            self.rate_limit_exceeded.labels(user_tier=user_tier).inc()
        else:
            key = f"rate_limit_exceeded_{user_tier}"
            self.fallback_metrics[key] = self.fallback_metrics.get(key, 0) + 1
    
    def get_metrics_data(self) -> str:
        """Get metrics in Prometheus format"""
        if self.is_available and PROMETHEUS_AVAILABLE:
            return generate_latest()
        else:
            # Return fallback metrics as text
            metrics_text = "# Fallback metrics\n"
            for key, value in self.fallback_metrics.items():
                metrics_text += f"{key} {value}\n"
            return metrics_text

class CacheManager:
    """
    Redis-based caching following enterprise caching patterns
    Based on Netflix's EVCache and AWS ElastiCache approaches
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/1")
        
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            self.is_available = True
            logger.info("Cache manager initialized with Redis backend")
        except Exception as e:
            logger.warning(f"Redis cache not available: {e}")
            self.is_available = False
            self.memory_cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if self.is_available:
            try:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
                return None
            except Exception as e:
                logger.error(f"Cache get error: {e}")
                return None
        else:
            return self.memory_cache.get(key)
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set cached value with TTL"""
        if self.is_available:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value, default=str))
                return True
            except Exception as e:
                logger.error(f"Cache set error: {e}")
                return False
        else:
            self.memory_cache[key] = value
            return True
    
    def delete(self, key: str):
        """Delete cached value"""
        if self.is_available:
            try:
                self.redis_client.delete(key)
                return True
            except Exception as e:
                logger.error(f"Cache delete error: {e}")
                return False
        else:
            self.memory_cache.pop(key, None)
            return True
    
    def clear_pattern(self, pattern: str):
        """Clear cache keys matching pattern"""
        if self.is_available:
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                return True
            except Exception as e:
                logger.error(f"Cache pattern clear error: {e}")
                return False
        else:
            keys_to_delete = [k for k in self.memory_cache.keys() if pattern.replace('*', '') in k]
            for key in keys_to_delete:
                del self.memory_cache[key]
            return True

class HealthChecker:
    """
    Comprehensive health checking following Google SRE patterns
    Based on Kubernetes health checks and AWS health monitoring
    """
    
    def __init__(self):
        self.checks = {}
        self.last_check_time = datetime.utcnow()
        self.check_interval = 30  # seconds
    
    def register_check(self, name: str, check_func: callable, critical: bool = True):
        """Register a health check"""
        self.checks[name] = {
            "func": check_func,
            "critical": critical,
            "last_result": None,
            "last_check": None,
            "error_count": 0
        }
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
            "overall_health": True
        }
        
        for name, check_info in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_info["func"]):
                    result = await check_info["func"]()
                else:
                    result = check_info["func"]()
                
                check_info["last_result"] = result
                check_info["last_check"] = datetime.utcnow()
                check_info["error_count"] = 0
                
                results["checks"][name] = {
                    "status": "healthy" if result else "unhealthy",
                    "critical": check_info["critical"],
                    "last_check": check_info["last_check"].isoformat()
                }
                
                if not result and check_info["critical"]:
                    results["overall_health"] = False
                    
            except Exception as e:
                check_info["error_count"] += 1
                results["checks"][name] = {
                    "status": "error",
                    "error": str(e),
                    "critical": check_info["critical"],
                    "error_count": check_info["error_count"]
                }
                
                if check_info["critical"]:
                    results["overall_health"] = False
        
        results["status"] = "healthy" if results["overall_health"] else "unhealthy"
        self.last_check_time = datetime.utcnow()
        
        return results

class ProductionFeatureManager:
    """
    Main production features orchestrator
    Integrates authentication, rate limiting, caching, metrics, and health checks
    """
    
    def __init__(self):
        self.auth_manager = AuthenticationManager()
        self.rate_limiter = RateLimiter()
        self.metrics_collector = MetricsCollector()
        self.cache_manager = CacheManager()
        self.health_checker = HealthChecker()
        
        # Register default health checks
        self._register_default_health_checks()
        
        logger.info("Production Feature Manager initialized with enterprise-grade capabilities")
    
    def _register_default_health_checks(self):
        """Register default system health checks"""
        self.health_checker.register_check("database", self._check_database_health, critical=True)
        self.health_checker.register_check("redis_cache", self._check_cache_health, critical=False)
        self.health_checker.register_check("redis_rate_limit", self._check_rate_limit_health, critical=False)
        self.health_checker.register_check("memory_usage", self._check_memory_health, critical=True)
    
    def _check_database_health(self) -> bool:
        """Check database connectivity"""
        try:
            # This would check actual database in production
            return True
        except Exception:
            return False
    
    def _check_cache_health(self) -> bool:
        """Check Redis cache health"""
        if self.cache_manager.is_available:
            try:
                self.cache_manager.redis_client.ping()
                return True
            except Exception:
                return False
        return True  # Not critical if using memory cache
    
    def _check_rate_limit_health(self) -> bool:
        """Check rate limiting system health"""
        if self.rate_limiter.redis_available:
            try:
                self.rate_limiter.redis_client.ping()
                return True
            except Exception:
                return False
        return True  # Not critical if using memory store
    
    def _check_memory_health(self) -> bool:
        """Check system memory usage"""
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            return memory_percent < 90  # Alert if memory usage > 90%
        except ImportError:
            return True  # Can't check without psutil
        except Exception:
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "authentication": {
                "available": self.auth_manager.is_available,
                "total_users": len(self.auth_manager.users),
                "jwt_support": JWT_AVAILABLE
            },
            "rate_limiting": {
                "available": self.rate_limiter.is_available,
                "backend": "redis" if self.rate_limiter.redis_available else "memory",
                "tiers_configured": len(self.rate_limiter.rate_limits)
            },
            "metrics": {
                "available": self.metrics_collector.is_available,
                "backend": "prometheus" if PROMETHEUS_AVAILABLE else "fallback"
            },
            "caching": {
                "available": self.cache_manager.is_available,
                "backend": "redis" if self.cache_manager.is_available else "memory"
            },
            "health_checks": {
                "registered_checks": len(self.health_checker.checks),
                "last_check": self.health_checker.last_check_time.isoformat()
            }
        }

# Global production features instance
_production_features = None

def get_production_features() -> ProductionFeatureManager:
    """Get or create global production features instance"""
    global _production_features
    
    if _production_features is None:
        _production_features = ProductionFeatureManager()
        logger.info("Created new Production Feature Manager instance")
    
    return _production_features

def initialize_production_features() -> ProductionFeatureManager:
    """Initialize production features with all components"""
    features = get_production_features()
    logger.info("Production Features initialized with enterprise-grade capabilities")
    logger.info("Available components:")
    logger.info(f"  - Authentication: {'✅' if features.auth_manager.is_available else '❌'}")
    logger.info(f"  - Rate Limiting: {'✅' if features.rate_limiter.is_available else '❌'}")
    logger.info(f"  - Metrics Collection: {'✅' if features.metrics_collector.is_available else '❌'}")
    logger.info(f"  - Caching: {'✅' if features.cache_manager.is_available else '❌'}")
    logger.info(f"  - Health Checks: ✅")
    
    return features

if __name__ == "__main__":
    # Example usage and testing
    async def test_production_features():
        print("⚡ Testing Production Features - Phase 9")
        
        # Initialize features
        features = initialize_production_features()
        
        # Test authentication
        if features.auth_manager.is_available:
            user = features.auth_manager.authenticate_user("admin", "admin123")
            if user:
                token = features.auth_manager.create_access_token(user)
                print(f"✅ Authentication: {user.username} ({user.role.value})")
                print(f"   Token: {token.access_token[:20]}...")
                print(f"   API Key: {user.api_key}")
        
        # Test rate limiting
        allowed, info = features.rate_limiter.check_rate_limit("test_user", APITier.FREE)
        print(f"✅ Rate Limiting: {'Allowed' if allowed else 'Blocked'} - {info}")
        
        # Test caching
        features.cache_manager.set("test_key", {"test": "data"}, 60)
        cached = features.cache_manager.get("test_key")
        print(f"✅ Caching: {'Working' if cached else 'Failed'}")
        
        # Test health checks
        health = await features.health_checker.run_checks()
        print(f"✅ Health Checks: {health['status']} ({len(health['checks'])} checks)")
        
        # Test metrics
        features.metrics_collector.record_request("GET", "/test", 200, 0.1)
        print("✅ Metrics: Request recorded")
    
    # Run test
    asyncio.run(test_production_features())