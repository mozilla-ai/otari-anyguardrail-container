FROM debian:bookworm-slim

WORKDIR /app

COPY granite-guardian-4.1-8b.Q6_K.llamafile ./granite-guardian-4.1-8b.Q6_K.llamafile

RUN chmod +x /app/granite-guardian-4.1-8b.Q6_K.llamafile

EXPOSE 8080

CMD ["sh", "-c", "cd /app && ./granite-guardian-4.1-8b.Q6_K.llamafile --host 0.0.0.0"]