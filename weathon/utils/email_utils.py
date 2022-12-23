# -*- coding: utf-8 -*-
# @Time    : 2022/12/22 22:11
# @Author  : LiZhen
# @FileName: email_utils.py
# @github  : https://github.com/Lizhen0628
# @Description:

import os
from typing import List
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class EmailUtils:
    """
    前置操作：
    1. 登陆163邮箱，进入设置，选择POP3/SMTP/IMAP
    2. 开启POP3/SMTP服务
    3. 手机验证
    """

    @staticmethod
    def send_email(from_user, to_users: str, subject: str = "", content: str = "", filenames: List[str] = None):
        """
        发送邮件
        Args:
            from_user: 发件人
            to_users: 收件人， 多个收件人用英文分号进行分割
            subject: 邮件主题
            content: 邮件正文内容
            filenames: 附件，要发送的文件路径

        Returns:

        """
        email = MIMEMultipart()
        email['From'] = from_user
        email['To'] = to_users
        email['Subject'] = subject

        message = MIMEText(content)
        email.attach(message)

        for filename in filenames:
            display_filename = os.path.basename(filename)
            fp = open(filename, 'rb')
            attachment = MIMEApplication(fp.read())
            attachment.add_header('Content-Disposition', 'attachment', filename=('utf-8', '', display_filename))
            email.attach(attachment)
            fp.close()

        email_host = 'smtp.163.com'  # 邮件服务器域名(自行修改)
        email_port = 465  # 邮件服务端口(通常是465)

        smtp = smtplib.SMTP_SSL(email_host, email_port)  # 创建SMTP_SSL对象(连接邮件服务器)
        smtp.login(from_user, '邮件服务器的授权码')  # 通过用户名和授权码进行登录  TODO: 类注释->根据前置操作获取邮件授权码
        smtp.sendmail(from_user, to_users.split(';'), email.as_string())  # 发送邮件(发件人，收件人，邮件内容)


if __name__ == '__main__':
    content_str = """在前面的课程中，我们已经"""

    filenames = ["学生成绩.xlsx"]
    EmailUtils.send_email(from_user='16621660628@163.com', to_users="188014193@qq.com;15725723275@163.com",
                          subject="邮件测试", content=content_str, filenames=filenames)
