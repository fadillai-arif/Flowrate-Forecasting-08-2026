import "./globals.css";

export const metadata = {
  title: "Flowrate Forecasting",
  description: "Water source flowrate forecasting application"
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
