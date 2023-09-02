# 使用 Python 3.9 作為基礎映像
FROM python:3.9
ENV workdir /advertorial-classifier

# 設定工作目錄
WORKDIR ${workdir}

# 將主要應用程式代碼複製到容器中
COPY . ${workdir}/

# 安裝所需的相依套件
RUN pip install -r requirements.txt && \
    chmod +x /${workdir}/train_and_summary.sh


# 定義容器的啟動指令
CMD [${workdir}"/train_and_summary.sh"]
# 暴露容器內部的 8080 埠
# EXPOSE 8080

# 定義容器的啟動指令
# ENTRYPOINT ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8080"]
