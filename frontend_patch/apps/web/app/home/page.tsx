import { PageBody, PageHeader } from '@kit/ui/page';

import { DashboardPage } from '~/home/dashboard/dashboard-page';

export default function HomePage() {
  return (
    <>
      <PageHeader description={'Your SaaS at a glance'} />

      <PageBody>
        <DashboardPage />
      </PageBody>
    </>
  );
}
