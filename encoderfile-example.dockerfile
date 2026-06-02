FROM debian:bookworm-slim

WORKDIR /app

COPY DuoGuard-0.5B.aarch64-linux-gnu.encoderfile ./DuoGuard-0.5B.aarch64-linux-gnu.encoderfile

RUN chmod +x /app/DuoGuard-0.5B.aarch64-linux-gnu.encoderfile

EXPOSE 8080

CMD ["/app/DuoGuard-0.5B.aarch64-linux-gnu.encoderfile", "serve"]