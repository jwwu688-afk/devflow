"""令牌加解密工具：GitHub PAT 等敏感凭证入库前加密，读取时解密。

业务背景：仓库接入表需要保存用户的 GitHub Token，明文入库一旦库被拖走等于凭证全泄漏，
因此用对称加密（Fernet = AES-CBC + HMAC 防篡改）存储，密钥由 settings 提供。
"""

from cryptography.fernet import Fernet, InvalidToken


def _derive_demo_key(secret: str) -> bytes:
    """把任意长度的配置字符串派生成 Fernet 可接受的 32 字节 base64 密钥。

    做法：SHA-256 摘要（固定 32 字节）→ urlsafe base64（Fernet 要求的编码）。
    为什么不用 KDF（PBKDF2/Argon2）：这里 secret 来自运维配置而非用户口令，
    没有暴力破解的压力，确定性派生还便于多实例共享同一密钥。
    """
    # Fernet 需要 urlsafe base64 编码的 32 字节密钥。
    # 这里的确定性本地开发派生方式适合课程项目，不适合生产环境。
    import base64
    import hashlib

    return base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())


def encrypt_token(token: str, secret: str) -> str:
    """加密明文 token，返回可直接存入 varchar 列的 base64 字符串。

    Fernet 每次加密都会带随机 IV，所以同一明文两次加密结果不同，这是预期行为。
    """
    return Fernet(_derive_demo_key(secret)).encrypt(token.encode("utf-8")).decode("utf-8")


def decrypt_token(encrypted: str, secret: str) -> str | None:
    """解密 token；密钥不匹配或密文被篡改时返回 None 而不是抛异常。

    返回 None 让上层能优雅降级（例如提示用户重新授权），
    避免一条脏数据把整个列表接口打成 500。
    """
    try:
        return Fernet(_derive_demo_key(secret)).decrypt(encrypted.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        # 解密失败最常见的原因：换了 token_encryption_key 而旧密文还在库里
        return None
