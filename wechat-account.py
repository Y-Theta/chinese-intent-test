import os
from flask import Flask, request, abort
from wechatpy import parse_message, create_reply
from wechatpy.utils import check_signature
from wechatpy.crypto import WeChatCrypto
from wechatpy.exceptions import InvalidSignatureException, InvalidAppIdException
from wechatpy import WeChatClient

class WeChatIntegratedServer:
    def __init__(self, token, app_id, app_secret, encoding_aes_key):
        """
        初始化微信集成服务器
        :param token: 微信后台配置的 Token
        :param app_id: 微信 AppID
        :param app_secret: 微信 AppSecret (用于主动调接口)
        :param encoding_aes_key: 微信后台配置的 EncodingAESKey
        """
        self.token = token
        self.app_id = app_id
        self.app_secret = app_secret
        self.encoding_aes_key = encoding_aes_key

        # 1. 初始化加密解密器 (用于安全模式)
        self.crypto = WeChatCrypto(token, encoding_aes_key, app_id)
        
        # 2. 初始化 API 客户端 (用于主动发送消息、获取用户信息等)
        self.client = WeChatClient(app_id, app_secret)

        # 3. 初始化 Flask 应用
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        """配置路由"""
        @self.app.route('/wechat', methods=['GET', 'POST'])
        def wechat_entry():
            # 获取通用参数
            signature = request.args.get('signature', '')
            timestamp = request.args.get('timestamp', '')
            nonce = request.args.get('nonce', '')

            # 安全校验：验证请求是否来自微信
            try:
                check_signature(self.token, signature, timestamp, nonce)
            except InvalidSignatureException:
                abort(403)

            # 处理首次接入验证 (GET)
            if request.method == 'GET':
                return request.args.get('echostr', '')

            # 处理用户消息推送到 (POST)
            msg_signature = request.args.get('msg_signature', '')
            return self._handle_message(request.data, msg_signature, timestamp, nonce)

    def _handle_message(self, raw_xml, msg_signature, timestamp, nonce):
        """处理解析、逻辑分发、加密返回"""
        try:
            # A. 解密收到的消息
            decrypted_xml = self.crypto.decrypt_message(
                raw_xml, msg_signature, timestamp, nonce
            )
            msg = parse_message(decrypted_xml)
            
            # B. 业务逻辑分发
            reply_text = self._logic_router(msg)

            # C. 构造响应对象
            reply = create_reply(reply_text, message=msg)
            
            # D. 加密响应内容并返回
            return self.crypto.encrypt_message(reply.render(), nonce, timestamp)

        except (InvalidSignatureException, InvalidAppIdException):
            abort(403)
        except Exception as e:
            self.app.logger.error(f"处理消息时发生错误: {e}")
            return "success"

    def _logic_router(self, msg):
        """核心业务逻辑：根据消息类型返回不同文字"""
        
        # 1. 处理语音消息 (需在后台开启语音识别)
        if msg.type == 'voice':
            # 获取微信识别好的文字内容
            recognition = getattr(msg, 'recognition', None)
            if recognition:
                return f"【语音转文字成功】\n内容：{recognition}"
            else:
                return "收到语音，但识别结果为空，请确保后台开启了『语音识别』功能。"

        # 2. 处理文字消息
        elif msg.type == 'text':
            content = msg.content.strip()
            if content == "你好":
                return "你好！我是你的 AI 助手。"
            return f"你刚才发的是文字：{content}"

        # 3. 处理关注事件
        elif msg.type == 'event' and msg.event == 'subscribe':
            return "欢迎关注本公众号！发送语音我可以帮你转成文字哦。"

        return "收到不支持的消息类型"

    def run(self, host='0.0.0.0', port=80):
        """启动 Flask 服务"""
        # 注意：生产环境下建议使用 gunicorn 或 uWSGI
        self.app.run(host=host, port=port, debug=False)

# --- 启动入口 ---
if __name__ == '__main__':
    # 建议将这些敏感信息放在环境变量中
    config = {
        'token': os.getenv('WECHAT_TOKEN', 'your_token'),
        'app_id': os.getenv('WECHAT_APPID', 'your_appid'),
        'app_secret': os.getenv('WECHAT_SECRET', 'your_secret'),
        'encoding_aes_key': os.getenv('WECHAT_AESKEY', 'your_aes_key')
    }

    server = WeChatIntegratedServer(**config)
    server.run(port=80)